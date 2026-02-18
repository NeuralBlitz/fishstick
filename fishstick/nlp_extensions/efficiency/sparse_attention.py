"""
Sparse Attention Mechanisms

Implementation of various sparse attention patterns for efficient
transformer models. Includes sliding window, block, dilated, and
random attention patterns.
"""

import torch
from torch import nn, Tensor
import math
from typing import Optional, Tuple, List


class SparseAttention(nn.Module):
    """Base class for sparse attention mechanisms.

    Provides common functionality for sparse attention patterns.
    """

    def __init__(self, num_heads: int):
        super().__init__()
        self.num_heads = num_heads

    def _create_mask(
        self,
        seq_len: int,
        device: torch.device,
    ) -> Tensor:
        """Create attention mask for sparse pattern."""
        raise NotImplementedError


class SlidingWindowAttention(SparseAttention):
    """Sliding window attention (also known as local attention).

    Each token only attends to tokens within a fixed window size.
    Provides O(n * w) complexity where w is the window size.

    Attributes:
        window_size: Size of the sliding window
    """

    def __init__(
        self,
        num_heads: int,
        window_size: int = 512,
    ):
        super().__init__(num_heads)
        self.window_size = window_size

    def _create_mask(
        self,
        seq_len: int,
        device: torch.device,
    ) -> Tensor:
        """Create sliding window attention mask."""
        mask = torch.full((seq_len, seq_len), float("-inf"), device=device)

        for i in range(seq_len):
            start = max(0, i - self.window_size // 2)
            end = min(seq_len, i + self.window_size // 2 + 1)
            mask[i, start:end] = 0.0

        return mask

    def forward(
        self,
        q: Tensor,
        k: Tensor,
        v: Tensor,
        attention_mask: Optional[Tensor] = None,
    ) -> Tensor:
        """Forward pass with sliding window attention.

        Args:
            q: Query tensor [batch, heads, seq_len, head_dim]
            k: Key tensor [batch, heads, seq_len, head_dim]
            v: Value tensor [batch, heads, seq_len, head_dim]
            attention_mask: Optional additional mask

        Returns:
            Attention output [batch, heads, seq_len, head_dim]
        """
        batch_size, num_heads, seq_len, head_dim = q.shape

        attn_scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(head_dim)

        window_mask = self._create_mask(seq_len, q.device)
        attn_scores = attn_scores + window_mask

        if attention_mask is not None:
            attn_scores = attn_scores + attention_mask

        attn_probs = torch.softmax(attn_scores, dim=-1)

        return torch.matmul(attn_probs, v)


class BlockSparseAttention(SparseAttention):
    """Block sparse attention.

    Divides the sequence into blocks and only attends within blocks
    and across a limited number of neighboring blocks.

    Attributes:
        block_size: Size of each block
        num_blocks: Number of blocks to attend to
    """

    def __init__(
        self,
        num_heads: int,
        block_size: int = 64,
        num_blocks: int = 3,
    ):
        super().__init__(num_heads)
        self.block_size = block_size
        self.num_blocks = num_blocks

    def _create_mask(
        self,
        seq_len: int,
        device: torch.device,
    ) -> Tensor:
        """Create block sparse attention mask."""
        num_blocks = (seq_len + self.block_size - 1) // self.block_size

        mask = torch.full((num_blocks, num_blocks), float("-inf"), device=device)

        for i in range(num_blocks):
            start = max(0, i - self.num_blocks // 2)
            end = min(num_blocks, i + self.num_blocks // 2 + 1)
            mask[i, start:end] = 0.0

        mask = mask.repeat_interleave(self.block_size, dim=0).repeat_interleave(
            self.block_size, dim=1
        )

        mask = mask[:seq_len, :seq_len]

        return mask

    def forward(
        self,
        q: Tensor,
        k: Tensor,
        v: Tensor,
        attention_mask: Optional[Tensor] = None,
    ) -> Tensor:
        """Forward pass with block sparse attention."""
        batch_size, num_heads, seq_len, head_dim = q.shape

        attn_scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(head_dim)

        block_mask = self._create_mask(seq_len, q.device)
        attn_scores = attn_scores + block_mask

        if attention_mask is not None:
            attn_scores = attn_scores + attention_mask

        attn_probs = torch.softmax(attn_scores, dim=-1)

        return torch.matmul(attn_probs, v)


class DilatedAttention(SparseAttention):
    """Dilated attention.

    Uses dilated convolution-style attention with gaps between
    attended positions for larger receptive field.

    Attributes:
        dilation_rate: Dilation rate for attention
        num_heads: Number of attention heads
    """

    def __init__(
        self,
        num_heads: int,
        dilation_rate: int = 2,
        window_size: int = 128,
    ):
        super().__init__(num_heads)
        self.dilation_rate = dilation_rate
        self.window_size = window_size

    def forward(
        self,
        q: Tensor,
        k: Tensor,
        v: Tensor,
        attention_mask: Optional[Tensor] = None,
    ) -> Tensor:
        """Forward pass with dilated attention."""
        batch_size, num_heads, seq_len, head_dim = q.shape

        attn_output = torch.zeros_like(q)

        for offset in range(self.dilation_rate):
            indices = torch.arange(offset, seq_len, self.dilation_rate, device=q.device)

            if len(indices) == 0:
                continue

            q_local = q[:, :, indices, :]
            k_local = k[:, :, indices, :]
            v_local = v[:, :, indices, :]

            attn_scores = torch.matmul(q_local, k_local.transpose(-2, -1)) / math.sqrt(
                head_dim
            )

            window = min(self.window_size, len(indices))
            local_mask = torch.full(
                (len(indices), len(indices)),
                float("-inf"),
                device=q.device,
            )
            for i in range(len(indices)):
                start = max(0, i - window // 2)
                end = min(len(indices), i + window // 2 + 1)
                local_mask[i, start:end] = 0.0

            attn_scores = attn_scores + local_mask

            attn_probs = torch.softmax(attn_scores, dim=-1)

            attn_output[:, :, indices, :] = torch.matmul(attn_probs, v_local)

        return attn_output


class GlobalLocalAttention(SparseAttention):
    """Global-local attention combining global and local patterns.

    Uses a mix of global tokens that attend to everything and local
    tokens that only attend within their window.

    Attributes:
        num_global_tokens: Number of global tokens
        window_size: Local attention window size
    """

    def __init__(
        self,
        num_heads: int,
        num_global_tokens: int = 1,
        window_size: int = 512,
    ):
        super().__init__(num_heads)
        self.num_global_tokens = num_global_tokens
        self.window_size = window_size

    def _create_mask(
        self,
        seq_len: int,
        device: torch.device,
    ) -> Tensor:
        """Create global-local attention mask."""
        mask = torch.zeros(seq_len, seq_len, device=device)

        for i in range(self.num_global_tokens, seq_len):
            start = max(0, i - self.window_size // 2)
            end = min(seq_len, i + self.window_size // 2 + 1)
            mask[i, start:end] = 0
            mask[i, : self.num_global_tokens] = 0

        mask[: self.num_global_tokens, :] = 0

        mask = mask.masked_fill(mask == 0, float("-inf"))
        mask = mask.masked_fill(mask > 0, 0)

        return mask

    def forward(
        self,
        q: Tensor,
        k: Tensor,
        v: Tensor,
        attention_mask: Optional[Tensor] = None,
    ) -> Tensor:
        """Forward pass with global-local attention."""
        batch_size, num_heads, seq_len, head_dim = q.shape

        attn_scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(head_dim)

        gl_mask = self._create_mask(seq_len, q.device)
        attn_scores = attn_scores + gl_mask

        if attention_mask is not None:
            attn_scores = attn_scores + attention_mask

        attn_probs = torch.softmax(attn_scores, dim=-1)

        return torch.matmul(attn_probs, v)


class RandomAttention(SparseAttention):
    """Random attention pattern.

    Each token attends to a fixed number of randomly selected
    other tokens, providing diversity in attention patterns.

    Attributes:
        num_random: Number of random connections per token
    """

    def __init__(
        self,
        num_heads: int,
        num_random: int = 32,
    ):
        super().__init__(num_heads)
        self.num_random = num_random

    def _create_random_indices(
        self,
        seq_len: int,
        device: torch.device,
    ) -> Tensor:
        """Create random attention indices."""
        indices = torch.randint(0, seq_len, (seq_len, self.num_random), device=device)
        return indices

    def forward(
        self,
        q: Tensor,
        k: Tensor,
        v: Tensor,
        attention_mask: Optional[Tensor] = None,
    ) -> Tensor:
        """Forward pass with random attention."""
        batch_size, num_heads, seq_len, head_dim = q.shape

        k_expanded = k.repeat_interleave(seq_len, dim=2)
        v_expanded = v.repeat_interleave(seq_len, dim=2)

        random_indices = self._create_random_indices(seq_len, q.device)

        attn_scores = torch.gather(
            torch.matmul(q, k.transpose(-2, -1)),
            -1,
            random_indices.unsqueeze(0)
            .unsqueeze(0)
            .expand(batch_size, num_heads, seq_len, -1),
        ) / math.sqrt(head_dim)

        attn_probs = torch.softmax(attn_scores, dim=-1)

        k_random = torch.gather(
            k_expanded.view(batch_size, num_heads, seq_len, seq_len, -1),
            3,
            random_indices.unsqueeze(0)
            .unsqueeze(0)
            .unsqueeze(-1)
            .expand(batch_size, num_heads, seq_len, self.num_random, head_dim),
        )

        v_random = torch.gather(
            v_expanded.view(batch_size, num_heads, seq_len, seq_len, -1),
            3,
            random_indices.unsqueeze(0)
            .unsqueeze(0)
            .unsqueeze(-1)
            .expand(batch_size, num_heads, seq_len, self.num_random, head_dim),
        )

        return torch.sum(attn_probs.unsqueeze(-1) * v_random, dim=3)


class SparseMultiHeadAttention(nn.Module):
    """Complete sparse multi-head attention module.

    Combines sparse attention patterns with standard multi-head
    attention interface.

    Attributes:
        embed_dim: Embedding dimension
        num_heads: Number of attention heads
        sparse_type: Type of sparse attention
        dropout: Dropout probability
    """

    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        sparse_type: str = "sliding_window",
        dropout: float = 0.0,
        **sparse_kwargs,
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

        if sparse_type == "sliding_window":
            self.sparse_attn = SlidingWindowAttention(
                num_heads,
                sparse_kwargs.get("window_size", 512),
            )
        elif sparse_type == "block":
            self.sparse_attn = BlockSparseAttention(
                num_heads,
                sparse_kwargs.get("block_size", 64),
                sparse_kwargs.get("num_blocks", 3),
            )
        elif sparse_type == "dilated":
            self.sparse_attn = DilatedAttention(
                num_heads,
                sparse_kwargs.get("dilation_rate", 2),
                sparse_kwargs.get("window_size", 128),
            )
        elif sparse_type == "global_local":
            self.sparse_attn = GlobalLocalAttention(
                num_heads,
                sparse_kwargs.get("num_global_tokens", 1),
                sparse_kwargs.get("window_size", 512),
            )
        elif sparse_type == "random":
            self.sparse_attn = RandomAttention(
                num_heads,
                sparse_kwargs.get("num_random", 32),
            )
        else:
            self.sparse_attn = SlidingWindowAttention(num_heads)

        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        x: Tensor,
        attention_mask: Optional[Tensor] = None,
    ) -> Tensor:
        """Forward pass with sparse attention.

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

        attn_output = self.sparse_attn(q, k, v, attention_mask)

        attn_output = (
            attn_output.transpose(1, 2)
            .contiguous()
            .view(batch_size, seq_len, self.embed_dim)
        )

        return self.out_proj(attn_output)


class BigBirdAttention(nn.Module):
    """BigBird-style attention combining multiple sparse patterns.

    Combines sliding window, global, and random attention
    as described in the BigBird paper.

    Attributes:
        embed_dim: Embedding dimension
        num_heads: Number of attention heads
        num_global: Number of global tokens
        num_random: Number of random connections
        window_size: Sliding window size
    """

    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        num_global: int = 2,
        num_random: int = 3,
        window_size: int = 3,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads

        self.num_global = num_global
        self.num_random = num_random
        self.window_size = window_size

        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        self.out_proj = nn.Linear(embed_dim, embed_dim)

    def forward(
        self,
        x: Tensor,
        attention_mask: Optional[Tensor] = None,
    ) -> Tensor:
        """Forward pass with BigBird-style attention."""
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

        mask = torch.zeros(seq_len, seq_len, device=x.device, dtype=torch.bool)

        for i in range(self.num_global):
            mask[i, :] = True
            mask[:, i] = True

        for i in range(seq_len):
            start = max(0, i - self.window_size)
            end = min(seq_len, i + self.window_size + 1)
            mask[i, start:end] = True

        random_indices = torch.randint(
            0, seq_len, (seq_len, self.num_random), device=x.device
        )
        for i in range(seq_len):
            mask[i, random_indices[i]] = True

        attn_scores = attn_scores.masked_fill(
            ~mask.unsqueeze(0).unsqueeze(0), float("-inf")
        )

        if attention_mask is not None:
            attn_scores = attn_scores + attention_mask

        attn_probs = torch.softmax(attn_scores, dim=-1)

        attn_output = torch.matmul(attn_probs, v)

        attn_output = (
            attn_output.transpose(1, 2)
            .contiguous()
            .view(batch_size, seq_len, self.embed_dim)
        )

        return self.out_proj(attn_output)
