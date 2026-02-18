"""
Set Transformer for Geometric Deep Learning.

Implements transformer architectures for processing unordered sets:
- Set attention blocks
- Induced set attention
- Pooling by multi-head attention
- Deep Sets

Based on:
- Lee et al. (2019): Set Transformer: A Framework for Attention-based
- Zaheer et al. (2017): Deep Sets
"""

from typing import Optional, Tuple, List
import torch
from torch import Tensor, nn
import torch.nn.functional as F
import math


class SetAttentionBlock(nn.Module):
    """
    Multi-head attention block for sets.

    Processes elements in a set while maintaining permutation invariance.
    """

    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads

        assert dim % num_heads == 0, "dim must be divisible by num_heads"

        self.q_proj = nn.Linear(dim, dim)
        self.k_proj = nn.Linear(dim, dim)
        self.v_proj = nn.Linear(dim, dim)

        self.out_proj = nn.Linear(dim, dim)

        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(dim)

    def forward(
        self,
        x: Tensor,
        mask: Optional[Tensor] = None,
    ) -> Tensor:
        """
        Apply set attention.

        Args:
            x: Input set [B, N, dim]
            mask: Optional attention mask [B, N]

        Returns:
            Attended features [B, N, dim]
        """
        B, N, _ = x.shape

        q = self.q_proj(x).view(B, N, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(B, N, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(B, N, self.num_heads, self.head_dim).transpose(1, 2)

        attn = (q @ k.transpose(-2, -1)) / math.sqrt(self.head_dim)

        if mask is not None:
            mask = mask.unsqueeze(1).unsqueeze(2)
            attn = attn.masked_fill(mask == 0, float("-inf"))

        attn = F.softmax(attn, dim=-1)
        attn = self.dropout(attn)

        out = attn @ v
        out = out.transpose(1, 2).contiguous().view(B, N, self.dim)

        out = self.out_proj(out)

        out = self.layer_norm(out + x)

        return out


class InducedSetAttentionBlock(nn.Module):
    """
    Induced Set Attention Block (ISAB).

    Uses induced points to reduce complexity from O(N^2) to O(NM).
    """

    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        num_induced: int = 16,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.num_induced = num_induced
        self.head_dim = dim // num_heads

        self.induced_points = nn.Parameter(torch.randn(1, num_induced, dim))

        self.q_proj = nn.Linear(dim, dim)
        self.k_proj = nn.Linear(dim, dim)
        self.v_proj = nn.Linear(dim, dim)

        self.out_proj = nn.Linear(dim, dim)

        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        x: Tensor,
        mask: Optional[Tensor] = None,
    ) -> Tensor:
        """
        Apply induced set attention.

        Args:
            x: Input set [B, N, dim]
            mask: Optional mask

        Returns:
            Attended features [B, N, dim]
        """
        B, N, _ = x.shape

        induced = self.induced_points.expand(B, -1, -1)

        q = self.q_proj(x).view(B, N, self.num_heads, self.head_dim).transpose(1, 2)
        k_ind = (
            self.k_proj(induced)
            .view(B, self.num_induced, self.num_heads, self.head_dim)
            .transpose(1, 2)
        )
        v_ind = (
            self.v_proj(induced)
            .view(B, self.num_induced, self.num_heads, self.head_dim)
            .transpose(1, 2)
        )

        attn_1 = (q @ k_ind.transpose(-2, -1)) / math.sqrt(self.head_dim)
        attn_1 = F.softmax(attn_1, dim=-1)
        attn_1 = self.dropout(attn_1)

        induced_out = attn_1 @ v_ind

        k = self.k_proj(x).view(B, N, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(B, N, self.num_heads, self.head_dim).transpose(1, 2)

        attn_2 = (induced_out @ k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        if mask is not None:
            mask = mask.unsqueeze(1).unsqueeze(2)
            attn_2 = attn_2.masked_fill(mask == 0, float("-inf"))

        attn_2 = F.softmax(attn_2, dim=-1)
        attn_2 = self.dropout(attn_2)

        out = attn_2 @ v
        out = out.transpose(1, 2).contiguous().view(B, N, self.dim)

        out = self.out_proj(out)

        return out


class PoolingByMultiHeadAttention(nn.Module):
    """
    PBMA: Pooling by Multi-Head Attention.

    Attention-based pooling for set-to-set transformation.
    """

    def __init__(
        self,
        dim: int,
        num_queries: int = 1,
        num_heads: int = 8,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.dim = dim
        self.num_queries = num_queries
        self.num_heads = num_heads
        self.head_dim = dim // num_heads

        self.query = nn.Parameter(torch.randn(1, num_queries, dim))

        self.k_proj = nn.Linear(dim, dim)
        self.v_proj = nn.Linear(dim, dim)

        self.out_proj = nn.Linear(dim, dim)

        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        x: Tensor,
        mask: Optional[Tensor] = None,
    ) -> Tuple[Tensor, Tensor]:
        """
        Pool set to fixed size.

        Args:
            x: Input set [B, N, dim]
            mask: Optional mask

        Returns:
            pooled: Pooled representation [B, num_queries, dim]
            weights: Attention weights [B, num_queries, N]
        """
        B, N, _ = x.shape

        queries = self.query.expand(B, -1, -1)

        q = queries.view(B, self.num_queries, self.num_heads, self.head_dim).transpose(
            1, 2
        )
        k = self.k_proj(x).view(B, N, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(B, N, self.num_heads, self.head_dim).transpose(1, 2)

        attn = (q @ k.transpose(-2, -1)) / math.sqrt(self.head_dim)

        if mask is not None:
            mask = mask.unsqueeze(1).unsqueeze(2)
            attn = attn.masked_fill(mask == 0, float("-inf"))

        attn = F.softmax(attn, dim=-1)
        attn = self.dropout(attn)

        out = attn @ v
        out = out.transpose(1, 2).contiguous().view(B, self.num_queries, self.dim)

        out = self.out_proj(out)

        return out, attn


class SetTransformer(nn.Module):
    """
    Complete Set Transformer architecture.
    """

    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        num_layers: int = 3,
        num_induced: int = 16,
        dropout: float = 0.0,
        use_isa: bool = True,
    ):
        super().__init__()

        self.use_isa = use_isa

        if use_isa:
            self.layers = nn.ModuleList(
                [
                    InducedSetAttentionBlock(dim, num_heads, num_induced, dropout)
                    for _ in range(num_layers)
                ]
            )
        else:
            self.layers = nn.ModuleList(
                [SetAttentionBlock(dim, num_heads, dropout) for _ in range(num_layers)]
            )

        self.pooling = PoolingByMultiHeadAttention(dim, num_heads=1, dropout=dropout)

    def forward(
        self,
        x: Tensor,
        mask: Optional[Tensor] = None,
    ) -> Tuple[Tensor, Tensor]:
        """
        Process set.

        Args:
            x: Input set [B, N, dim]
            mask: Optional mask

        Returns:
            pooled: Pooled representation [B, 1, dim]
            attended: Full sequence [B, N, dim]
        """
        for layer in self.layers:
            x = layer(x, mask)

        pooled, _ = self.pooling(x, mask)

        return pooled, x


class DeepSet(nn.Module):
    """
    Deep Sets architecture for permutation-invariant processing.

    Based on Zaheer et al. (2017).
    """

    def __init__(
        self,
        in_dim: int,
        hidden_dim: int,
        out_dim: int,
        num_layers: int = 3,
        pooling: str = "mean",
    ):
        super().__init__()
        self.pooling = pooling

        encoder_layers = []
        encoder_layers.append(nn.Linear(in_dim, hidden_dim))
        encoder_layers.append(nn.SiLU())

        for _ in range(num_layers - 2):
            encoder_layers.append(nn.Linear(hidden_dim, hidden_dim))
            encoder_layers.append(nn.SiLU())

        self.encoder = nn.Sequential(*encoder_layers)

        decoder_layers = []
        decoder_layers.append(nn.Linear(hidden_dim, hidden_dim))
        decoder_layers.append(nn.SiLU())

        for _ in range(num_layers - 2):
            decoder_layers.append(nn.Linear(hidden_dim, hidden_dim))
            decoder_layers.append(nn.SiLU())

        decoder_layers.append(nn.Linear(hidden_dim, out_dim))

        self.decoder = nn.Sequential(*decoder_layers)

    def forward(
        self,
        x: Tensor,
        mask: Optional[Tensor] = None,
    ) -> Tensor:
        """
        Apply Deep Sets.

        Args:
            x: Input set [B, N, in_dim]
            mask: Optional mask

        Returns:
            Output [B, out_dim]
        """
        encoded = self.encoder(x)

        if mask is not None:
            masked = encoded * mask.unsqueeze(-1)
            if self.pooling == "mean":
                pooled = masked.sum(dim=1) / mask.sum(dim=1, keepdim=True)
            elif self.pooling == "sum":
                pooled = masked.sum(dim=1)
            else:
                pooled = masked.max(dim=1)[0]
        else:
            if self.pooling == "mean":
                pooled = encoded.mean(dim=1)
            elif self.pooling == "sum":
                pooled = encoded.sum(dim=1)
            else:
                pooled = encoded.max(dim=1)[0]

        return self.decoder(pooled)


class SetEncoder(nn.Module):
    """
    Encoder for unordered sets.
    """

    def __init__(
        self,
        in_dim: int,
        hidden_dim: int,
        out_dim: int,
        num_layers: int = 3,
        dropout: float = 0.0,
    ):
        super().__init__()

        self.embedding = nn.Linear(in_dim, hidden_dim)

        self.layers = nn.ModuleList(
            [SetAttentionBlock(hidden_dim, dropout=dropout) for _ in range(num_layers)]
        )

        self.pooling = PoolingByMultiHeadAttention(
            hidden_dim, num_heads=1, dropout=dropout
        )

        self.output = nn.Linear(hidden_dim, out_dim)

    def forward(
        self,
        x: Tensor,
        mask: Optional[Tensor] = None,
    ) -> Tensor:
        """
        Encode set to fixed-size representation.

        Args:
            x: Input set [B, N, in_dim]
            mask: Optional mask

        Returns:
            Fixed-size encoding [B, out_dim]
        """
        x = self.embedding(x)

        for layer in self.layers:
            x = layer(x, mask)

        pooled, _ = self.pooling(x, mask)

        return self.output(pooled).squeeze(1)


class SetResBlock(nn.Module):
    """
    Residual block for set processing.
    """

    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        dropout: float = 0.0,
    ):
        super().__init__()

        self.attention = SetAttentionBlock(dim, num_heads, dropout)
        self.ffn = nn.Sequential(
            nn.Linear(dim, dim * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim * 4, dim),
            nn.Dropout(dropout),
        )

        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)

    def forward(
        self,
        x: Tensor,
        mask: Optional[Tensor] = None,
    ) -> Tensor:
        """Forward with residual connections."""
        x = self.norm1(x + self.attention(x, mask))
        x = self.norm2(x + self.ffn(x))

        return x


class Set2SetPool(nn.Module):
    """
    Set2Set pooling operation.
    """

    def __init__(
        self,
        dim: int,
        num_steps: int = 3,
    ):
        super().__init__()
        self.dim = dim
        self.num_steps = num_steps

        self.q_proj = nn.Linear(dim, dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim * 2, dim),
            nn.Tanh(),
        )

    def forward(
        self,
        x: Tensor,
        mask: Optional[Tensor] = None,
    ) -> Tensor:
        """
        Set2Set pooling.

        Args:
            x: Input set [B, N, dim]
            mask: Optional mask

        Returns:
            Pooled representation [B, dim]
        """
        B, N, _ = x.shape

        q = torch.zeros(B, 1, self.dim, device=x.device)

        for _ in range(self.num_steps):
            q_expanded = q.expand(-1, N, -1)

            attention = (x * q_expanded).sum(dim=-1, keepdim=True)

            if mask is not None:
                attention = attention.masked_fill(
                    mask.unsqueeze(-1) == 0, float("-inf")
                )

            attention = F.softmax(attention, dim=1)

            weighted = (x * attention).sum(dim=1, keepdim=True)

            q = self.mlp(torch.cat([q, weighted], dim=-1))

        return q.squeeze(1)


class MultiSetAttention(nn.Module):
    """
    Multi-head set attention with multiple output tokens.
    """

    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        num_seeds: int = 1,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.num_seeds = num_seeds
        self.head_dim = dim // num_heads

        self.seed = nn.Parameter(torch.randn(1, num_seeds, dim))

        self.q_proj = nn.Linear(dim, dim)
        self.k_proj = nn.Linear(dim, dim)
        self.v_proj = nn.Linear(dim, dim)

        self.out_proj = nn.Linear(dim, dim)

        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        x: Tensor,
        mask: Optional[Tensor] = None,
    ) -> Tensor:
        """
        Multi-seed attention pooling.

        Args:
            x: Input set [B, N, dim]
            mask: Optional mask

        Returns:
            Seeded outputs [B, num_seeds, dim]
        """
        B, N, _ = x.shape

        seeds = self.seed.expand(B, -1, -1)

        q = (
            self.q_proj(seeds)
            .view(B, self.num_seeds, self.num_heads, self.head_dim)
            .transpose(1, 2)
        )
        k = self.k_proj(x).view(B, N, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(B, N, self.num_heads, self.head_dim).transpose(1, 2)

        attn = (q @ k.transpose(-2, -1)) / math.sqrt(self.head_dim)

        if mask is not None:
            mask_expanded = mask.unsqueeze(1).unsqueeze(2)
            attn = attn.masked_fill(mask_expanded == 0, float("-inf"))

        attn = F.softmax(attn, dim=-1)
        attn = self.dropout(attn)

        out = attn @ v
        out = out.transpose(1, 2).contiguous().view(B, self.num_seeds, self.dim)

        return self.out_proj(out)


__all__ = [
    "SetAttentionBlock",
    "InducedSetAttentionBlock",
    "PoolingByMultiHeadAttention",
    "SetTransformer",
    "DeepSet",
    "SetEncoder",
    "SetResBlock",
    "Set2SetPool",
    "MultiSetAttention",
]
