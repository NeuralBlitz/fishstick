"""
Flash Attention Implementation

Memory-efficient attention mechanism using IO-aware attention algorithms.
Provides fast and memory-efficient attention computation.
"""

import torch
from torch import nn, Tensor
import math
from typing import Optional, Tuple
import torch.nn.functional as F


class FlashAttention(nn.Module):
    """Flash Attention implementation.

    Memory-efficient attention that computes attention in blocks
    to reduce memory usage from O(n^2) to O(n).

    Attributes:
        num_heads: Number of attention heads
        head_dim: Dimension of each head
        dropout: Dropout probability
    """

    def __init__(
        self,
        num_heads: int,
        head_dim: int,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.dropout = dropout
        self.scale = head_dim**-0.5

    def forward(
        self,
        q: Tensor,
        k: Tensor,
        v: Tensor,
        attention_mask: Optional[Tensor] = None,
        is_causal: bool = True,
    ) -> Tensor:
        """Forward pass with flash attention.

        Args:
            q: Query tensor [batch, heads, seq_len, head_dim]
            k: Key tensor [batch, heads, seq_len, head_dim]
            v: Value tensor [batch, heads, seq_len, head_dim]
            attention_mask: Optional attention mask
            is_causal: Whether to use causal masking

        Returns:
            Attention output [batch, heads, seq_len, head_dim]
        """
        try:
            from flash_attn import flash_attn_func

            if attention_mask is not None:
                return self._standard_attention(q, k, v, attention_mask, is_causal)

            output = flash_attn_func(
                q,
                k,
                v,
                dropout_p=self.dropout if self.training else 0.0,
                softmax_scale=self.scale,
                causal=is_causal,
            )

            return output

        except ImportError:
            return self._standard_attention(q, k, v, attention_mask, is_causal)

    def _standard_attention(
        self,
        q: Tensor,
        k: Tensor,
        v: Tensor,
        attention_mask: Optional[Tensor],
        is_causal: bool,
    ) -> Tensor:
        """Standard attention as fallback."""
        attn_scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale

        if is_causal:
            seq_len = q.shape[2]
            causal_mask = torch.triu(
                torch.ones(seq_len, seq_len, device=q.device, dtype=torch.bool),
                diagonal=1,
            )
            attn_scores = attn_scores.masked_fill(causal_mask, float("-inf"))

        if attention_mask is not None:
            attn_scores = attn_scores + attention_mask

        attn_probs = F.softmax(attn_scores, dim=-1)

        if self.dropout > 0 and self.training:
            attn_probs = F.dropout(attn_probs, p=self.dropout)

        return torch.matmul(attn_probs, v)


class FlashAttentionV2(nn.Module):
    """Flash Attention v2 implementation.

    Improved version with better performance on certain hardware.
    """

    def __init__(
        self,
        num_heads: int,
        head_dim: int,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.dropout = dropout
        self.scale = head_dim**-0.5

    def forward(
        self,
        q: Tensor,
        k: Tensor,
        v: Tensor,
        attention_mask: Optional[Tensor] = None,
        is_causal: bool = True,
    ) -> Tensor:
        """Forward pass with Flash Attention v2."""
        try:
            from flash_attn import flash_attn_varlen_func

            batch_size, num_heads, seq_len, head_dim = q.shape

            q = q.transpose(1, 2).contiguous()
            k = k.transpose(1, 2).contiguous()
            v = v.transpose(1, 2).contiguous()

            cu_seqlens = torch.arange(
                0,
                (batch_size + 1) * seq_len,
                seq_len,
                device=q.device,
                dtype=torch.int32,
            )

            output = flash_attn_varlen_func(
                q,
                k,
                v,
                cu_seqlens,
                cu_seqlens,
                max_seqlen=seq_len,
                dropout_p=self.dropout if self.training else 0.0,
                softmax_scale=self.scale,
                causal=is_causal,
            )

            return output.transpose(1, 2)

        except ImportError:
            return self._standard_fallback(q, k, v, attention_mask, is_causal)

    def _standard_fallback(
        self,
        q: Tensor,
        k: Tensor,
        v: Tensor,
        attention_mask: Optional[Tensor],
        is_causal: bool,
    ) -> Tensor:
        """Standard attention fallback."""
        attn_scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale

        if is_causal:
            seq_len = q.shape[2]
            causal_mask = torch.triu(
                torch.ones(seq_len, seq_len, device=q.device, dtype=torch.bool),
                diagonal=1,
            )
            attn_scores = attn_scores.masked_fill(causal_mask, float("-inf"))

        if attention_mask is not None:
            attn_scores = attn_scores + attention_mask

        attn_probs = F.softmax(attn_scores, dim=-1)

        return torch.matmul(attn_probs, v)


class MemoryEfficientAttention(nn.Module):
    """Memory-efficient attention with chunked computation.

    Computes attention in chunks to reduce memory usage
    without requiring specialized CUDA kernels.

    Attributes:
        num_heads: Number of attention heads
        head_dim: Dimension of each head
        chunk_size: Size of chunks for computation
    """

    def __init__(
        self,
        num_heads: int,
        head_dim: int,
        chunk_size: int = 1024,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.chunk_size = chunk_size
        self.dropout = dropout
        self.scale = head_dim**-0.5

    def forward(
        self,
        q: Tensor,
        k: Tensor,
        v: Tensor,
        attention_mask: Optional[Tensor] = None,
        is_causal: bool = True,
    ) -> Tensor:
        """Forward pass with memory-efficient chunked attention."""
        batch_size, num_heads, seq_len, head_dim = q.shape

        output = torch.zeros_like(q)

        for start_idx in range(0, seq_len, self.chunk_size):
            end_idx = min(start_idx + self.chunk_size, seq_len)

            q_chunk = q[:, :, start_idx:end_idx, :]

            if is_causal:
                key_chunk_end = end_idx
            else:
                key_chunk_end = seq_len

            attn_scores = torch.matmul(
                q_chunk, k[:, :, :key_chunk_end, :].transpose(-2, -1)
            )
            attn_scores = attn_scores * self.scale

            if is_causal:
                causal_mask = torch.triu(
                    torch.ones(
                        end_idx - start_idx,
                        key_chunk_end,
                        device=q.device,
                        dtype=torch.bool,
                    ),
                    diagonal=start_idx + 1 - start_idx,
                )
                attn_scores = attn_scores.masked_fill(causal_mask, float("-inf"))

            if attention_mask is not None:
                attn_scores = (
                    attn_scores
                    + attention_mask[:, :, start_idx:end_idx, :key_chunk_end]
                )

            attn_probs = F.softmax(attn_scores, dim=-1)

            if self.dropout > 0 and self.training:
                attn_probs = F.dropout(attn_probs, p=self.dropout)

            output[:, :, start_idx:end_idx, :] = torch.matmul(
                attn_probs,
                v[:, :, :key_chunk_end, :],
            )

        return output


class FlashMultiHeadAttention(nn.Module):
    """Complete multi-head attention with flash attention support.

    Combines flash attention with standard multi-head attention
    interface and projection layers.

    Attributes:
        embed_dim: Embedding dimension
        num_heads: Number of attention heads
        dropout: Dropout probability
        use_flash: Whether to use flash attention
    """

    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        dropout: float = 0.0,
        use_flash: bool = True,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.use_flash = use_flash

        assert embed_dim % num_heads == 0

        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        self.out_proj = nn.Linear(embed_dim, embed_dim)

        if use_flash:
            self.flash_attn = FlashAttention(num_heads, self.head_dim, dropout)
        else:
            self.flash_attn = None

        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        x: Tensor,
        attention_mask: Optional[Tensor] = None,
        is_causal: bool = True,
    ) -> Tensor:
        """Forward pass with flash attention.

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

        if self.use_flash and self.flash_attn is not None:
            attn_output = self.flash_attn(q, k, v, attention_mask, is_causal)
        else:
            attn_output = MemoryEfficientAttention(
                self.num_heads,
                self.head_dim,
                dropout=self.dropout.p,
            )(q, k, v, attention_mask, is_causal)

        attn_output = (
            attn_output.transpose(1, 2)
            .contiguous()
            .view(batch_size, seq_len, self.embed_dim)
        )

        return self.out_proj(attn_output)


class FlashAttentionWithPdrop(nn.Module):
    """Flash attention with probabilistic dropout.

    Implements the p-dropout technique from certain efficient
    transformer implementations.
    """

    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        p_dropout: float = 0.0,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.p_dropout = p_dropout

        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        self.out_proj = nn.Linear(embed_dim, embed_dim)

    def forward(
        self,
        x: Tensor,
        attention_mask: Optional[Tensor] = None,
    ) -> Tensor:
        """Forward pass with probabilistic dropout."""
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

        if attention_mask is not None:
            attn_scores = attn_scores + attention_mask

        attn_weights = F.softmax(attn_scores, dim=-1)

        if self.p_dropout > 0 and self.training:
            attn_weights = F.dropout(attn_weights, p=self.p_dropout)

        attn_output = torch.matmul(attn_weights, v)

        attn_output = (
            attn_output.transpose(1, 2)
            .contiguous()
            .view(batch_size, seq_len, self.embed_dim)
        )

        return self.out_proj(attn_output)


def is_flash_attn_available() -> bool:
    """Check if flash attention is available.

    Returns:
        True if flash attention is available, False otherwise
    """
    try:
        from flash_attn import flash_attn_func

        return True
    except ImportError:
        return False


def create_flash_attention(
    embed_dim: int,
    num_heads: int,
    dropout: float = 0.0,
) -> nn.Module:
    """Create the best available flash attention module.

    Args:
        embed_dim: Embedding dimension
        num_heads: Number of attention heads
        dropout: Dropout probability

    Returns:
        Flash attention module
    """
    if is_flash_attn_available():
        return FlashMultiHeadAttention(embed_dim, num_heads, dropout, use_flash=True)
    else:
        return FlashMultiHeadAttention(embed_dim, num_heads, dropout, use_flash=False)
