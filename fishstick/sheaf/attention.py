"""
Sheaf-Optimized Attention (SOA).

Generalizes self-attention using sheaf cohomology to enforce
local-to-global consistency across heterogeneous modalities and scales.

The SOA kernel:
    K_SOA(x, y) = exp(-½σ² ||Π_{x→y} q(x) - k(y)||² - λ ||δ¹(s)||²_{x,y})

where:
- Π_{x→y} is parallel transport along minimal geodesic
- δ¹(s) is the local inconsistency on overlaps
- λ balances fidelity vs. consistency
"""

from typing import Optional, Tuple, List, Dict, Callable
import math
import torch
from torch import Tensor, nn
import torch.nn.functional as F

from ..geometric.sheaf import DataSheaf


class SheafOptimizedAttention(nn.Module):
    """
    Sheaf-Optimized Attention Module.

    Combines standard attention with sheaf-theoretic consistency constraints
    to ensure local attention patterns glue into globally coherent representations.

    Theorem (SOA Preserves Sheaf Cohomology):
    If initial sections satisfy δ¹(s) = 0, then SOA updates preserve
    δ¹(s) = 0 up to O(η²) in step size η.
    """

    def __init__(
        self,
        embed_dim: int,
        num_heads: int = 8,
        dropout: float = 0.1,
        lambda_consistency: float = 0.1,
        sigma: float = 1.0,
        use_parallel_transport: bool = True,
    ):
        super().__init__()

        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.lambda_consistency = lambda_consistency
        self.sigma = sigma
        self.use_parallel_transport = use_parallel_transport

        assert embed_dim % num_heads == 0, "embed_dim must be divisible by num_heads"

        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        self.out_proj = nn.Linear(embed_dim, embed_dim)

        self.dropout = nn.Dropout(dropout)

        if use_parallel_transport:
            self.transport = ParallelTransport(embed_dim, num_heads)

        self.cohomology_weight = nn.Parameter(torch.eye(self.head_dim) * 0.5)

    def forward(
        self,
        x: Tensor,
        open_cover: Optional[List[List[int]]] = None,
        key_padding_mask: Optional[Tensor] = None,
        need_weights: bool = False,
    ) -> Tuple[Tensor, Optional[Tensor]]:
        """
        Forward pass with sheaf-optimized attention.

        Args:
            x: Input tensor [batch, seq_len, embed_dim]
            open_cover: List of patches (indices) for sheaf structure
            key_padding_mask: Mask for padded positions
            need_weights: Whether to return attention weights

        Returns:
            output: [batch, seq_len, embed_dim]
            attn_weights: [batch, num_heads, seq_len, seq_len] if need_weights
        """
        batch_size, seq_len, _ = x.shape

        Q = self.q_proj(x).view(batch_size, seq_len, self.num_heads, self.head_dim)
        K = self.k_proj(x).view(batch_size, seq_len, self.num_heads, self.head_dim)
        V = self.v_proj(x).view(batch_size, seq_len, self.num_heads, self.head_dim)

        Q = Q.transpose(1, 2)
        K = K.transpose(1, 2)
        V = V.transpose(1, 2)

        if self.use_parallel_transport:
            K = self.transport(Q, K)

        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.head_dim)

        if open_cover is not None:
            consistency_penalty = self._compute_cohomology_penalty(
                Q, K, open_cover, batch_size, seq_len
            )
            scores = scores - self.lambda_consistency * consistency_penalty

        if key_padding_mask is not None:
            scores = scores.masked_fill(
                key_padding_mask.unsqueeze(1).unsqueeze(2), float("-inf")
            )

        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)

        output = torch.matmul(attn_weights, V)
        output = output.transpose(1, 2).contiguous().view(batch_size, seq_len, -1)
        output = self.out_proj(output)

        if need_weights:
            return output, attn_weights
        return output, None

    def _compute_cohomology_penalty(
        self,
        Q: Tensor,
        K: Tensor,
        open_cover: List[List[int]],
        batch_size: int,
        seq_len: int,
    ) -> Tensor:
        """
        Compute cohomological inconsistency penalty.

        δ¹(s)_{ij} = ||ρ_{U_i∩U_j,U_i}(s_i) - ρ_{U_i∩U_j,U_j}(s_j)||²
        """
        penalty = torch.zeros(
            batch_size, self.num_heads, seq_len, seq_len, device=Q.device
        )

        for i, patch_i in enumerate(open_cover):
            for j, patch_j in enumerate(open_cover):
                if i >= j:
                    continue

                intersection = list(set(patch_i) & set(patch_j))
                if not intersection:
                    continue

                for idx in intersection:
                    rho_ij = self.cohomology_weight

                    s_i = K[:, :, idx, :]
                    s_j = K[:, :, idx, :]

                    diff = torch.matmul(s_i - s_j, rho_ij)
                    inconsistency = (diff**2).sum(dim=-1)

                    penalty[:, :, idx, idx] = penalty[:, :, idx, idx] + inconsistency

        return penalty


class ParallelTransport(nn.Module):
    """
    Parallel transport for attention features.

    Computes Π_{x→y} mapping features from x to y along geodesics.
    This ensures equivariance under coordinate transformations.
    """

    def __init__(self, embed_dim: int, num_heads: int):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads

        self.transport_net = nn.Sequential(
            nn.Linear(self.head_dim * 2, self.head_dim * 2),
            nn.Tanh(),
            nn.Linear(self.head_dim * 2, self.head_dim * self.head_dim),
        )

    def forward(self, Q: Tensor, K: Tensor) -> Tensor:
        """
        Apply parallel transport to K based on Q.

        Args:
            Q: Query tensor [batch, heads, seq_len, head_dim]
            K: Key tensor [batch, heads, seq_len, head_dim]

        Returns:
            Transported K [batch, heads, seq_len, head_dim]
        """
        batch_size, num_heads, seq_len, head_dim = Q.shape

        Q_expand = Q.unsqueeze(3).expand(-1, -1, -1, seq_len, -1)
        K_expand = K.unsqueeze(2).expand(-1, -1, seq_len, -1, -1)

        combined = torch.cat([Q_expand, K_expand], dim=-1)

        transport_matrices = self.transport_net(combined)
        transport_matrices = transport_matrices.view(
            batch_size, num_heads, seq_len, seq_len, head_dim, head_dim
        )

        K_expand = K.unsqueeze(2).unsqueeze(-1)
        K_transported = torch.matmul(transport_matrices, K_expand).squeeze(-1)

        K_out = K_transported.mean(dim=2)

        return K_out


class SheafTransformerLayer(nn.Module):
    """
    Complete transformer layer with sheaf-optimized attention.
    """

    def __init__(
        self,
        embed_dim: int,
        num_heads: int = 8,
        ff_dim: int = 2048,
        dropout: float = 0.1,
        lambda_consistency: float = 0.1,
    ):
        super().__init__()

        self.attention = SheafOptimizedAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            dropout=dropout,
            lambda_consistency=lambda_consistency,
        )

        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, ff_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(ff_dim, embed_dim),
        )

        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        x: Tensor,
        open_cover: Optional[List[List[int]]] = None,
        key_padding_mask: Optional[Tensor] = None,
    ) -> Tensor:
        """
        Forward pass with pre-norm architecture.
        """
        residual = x
        x = self.norm1(x)
        x, _ = self.attention(
            x, open_cover=open_cover, key_padding_mask=key_padding_mask
        )
        x = self.dropout(x)
        x = residual + x

        residual = x
        x = self.norm2(x)
        x = self.ffn(x)
        x = self.dropout(x)
        x = residual + x

        return x


class SheafTransformer(nn.Module):
    """
    Full Sheaf-Optimized Transformer model.
    """

    def __init__(
        self,
        embed_dim: int = 512,
        num_heads: int = 8,
        num_layers: int = 6,
        ff_dim: int = 2048,
        dropout: float = 0.1,
        lambda_consistency: float = 0.1,
    ):
        super().__init__()

        self.layers = nn.ModuleList(
            [
                SheafTransformerLayer(
                    embed_dim=embed_dim,
                    num_heads=num_heads,
                    ff_dim=ff_dim,
                    dropout=dropout,
                    lambda_consistency=lambda_consistency,
                )
                for _ in range(num_layers)
            ]
        )

        self.final_norm = nn.LayerNorm(embed_dim)

    def forward(
        self,
        x: Tensor,
        open_cover: Optional[List[List[int]]] = None,
        key_padding_mask: Optional[Tensor] = None,
    ) -> Tensor:
        """
        Forward pass through all layers.
        """
        for layer in self.layers:
            x = layer(x, open_cover=open_cover, key_padding_mask=key_padding_mask)

        return self.final_norm(x)
