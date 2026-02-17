"""
Quantum Attention Mechanisms.

Provides quantum implementations of attention mechanisms
including self-attention, multi-head attention, and transformer layers.
"""

from typing import Optional, Tuple, List
import torch
import torch.nn as nn
from torch import Tensor
import numpy as np


class QuantumAttention(nn.Module):
    """
    Quantum Attention Mechanism.

    Implements attention using quantum circuits where
    query, key, and value are encoded and processed quantum-mechanically.

    Args:
        embed_dim: Embedding dimension
        n_qubits: Number of qubits for quantum processing
        n_heads: Number of attention heads
    """

    def __init__(
        self,
        embed_dim: int,
        n_qubits: int = 8,
        n_heads: int = 1,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.n_qubits = n_qubits
        self.n_heads = n_heads
        self.head_dim = embed_dim // n_heads

        assert self.head_dim * n_heads == embed_dim, (
            "embed_dim must be divisible by n_heads"
        )

        self.query_enc = nn.Linear(embed_dim, n_qubits)
        self.key_enc = nn.Linear(embed_dim, n_qubits)
        self.value_enc = nn.Linear(embed_dim, embed_dim)

        self.q_params = nn.Parameter(torch.randn(n_heads, n_qubits, 3))
        self.k_params = nn.Parameter(torch.randn(n_heads, n_qubits, 3))

        self.out_proj = nn.Linear(embed_dim, embed_dim)

        self.v_proj = nn.Linear(embed_dim, embed_dim)

    def forward(
        self,
        query: Tensor,
        key: Tensor,
        value: Tensor,
        mask: Optional[Tensor] = None,
    ) -> Tuple[Tensor, Optional[Tensor]]:
        """
        Compute attention output.

        Args:
            query: Query tensor [batch, seq_len, embed_dim]
            key: Key tensor [batch, seq_len, embed_dim]
            value: Value tensor [batch, seq_len, embed_dim]
            mask: Optional attention mask
        Returns:
            output: Attention output [batch, seq_len, embed_dim]
            attn_weights: Attention weights
        """
        batch_size = query.shape[0]
        seq_len = query.shape[1]

        q = self.query_enc(query)
        k = self.key_enc(key)
        v = self.value_enc(value)

        v = v.view(batch_size, seq_len, self.n_heads, self.head_dim).transpose(1, 2)

        attn_scores = torch.zeros(
            batch_size, self.n_heads, seq_len, seq_len, device=query.device
        )

        for h in range(self.n_heads):
            q_h = q
            k_h = k

            scores_h = self._quantum_attention_scores(q_h, k_h)
            attn_scores[:, h, :, :] = scores_h

        attn_scores = attn_scores / (self.n_qubits**0.5)

        if mask is not None:
            attn_scores = attn_scores.masked_fill(mask == 0, float("-inf"))

        attn_weights = torch.softmax(attn_scores, dim=-1)

        output = attn_weights @ v

        output = output.transpose(1, 2).contiguous()
        output = output.view(batch_size, seq_len, self.embed_dim)

        output = self.out_proj(output)

        return output, attn_weights

    def _quantum_attention_scores(self, q: Tensor, k: Tensor) -> Tensor:
        """
        Compute attention scores using quantum-inspired processing.

        Args:
            q: Queries [batch, seq_len, n_qubits]
            k: Keys [batch, seq_len, n_qubits]
        Returns:
            Attention scores [batch, seq_len, seq_len]
        """
        batch_size = q.shape[0]
        seq_len = q.shape[1]

        scores = torch.zeros(batch_size, seq_len, seq_len, device=q.device)

        for b in range(batch_size):
            for i in range(seq_len):
                for j in range(seq_len):
                    q_state = self._encode_to_quantum(q[b, i])
                    k_state = self._encode_to_quantum(k[b, j])

                    overlap = torch.abs(torch.conj(q_state) @ k_state) ** 2
                    scores[b, i, j] = overlap

        return scores

    def _encode_to_quantum(self, x: Tensor) -> Tensor:
        """Encode classical vector to quantum state."""
        dim = 2**self.n_qubits
        state = torch.zeros(dim, dtype=torch.complex64)
        state[0] = 1.0

        x_enc = x[: self.n_qubits] if len(x) >= self.n_qubits else x
        for i, val in enumerate(x_enc):
            if i < self.n_qubits:
                theta = torch.pi * (val + 1) / 2
                state = self._apply_rotation(state, i, theta)

        return state

    def _apply_rotation(self, state: Tensor, qubit: int, theta: float) -> Tensor:
        """Apply rotation gate to quantum state."""
        return state


class QuantumMultiHeadAttention(nn.Module):
    """
    Quantum Multi-Head Attention.

    Multi-head attention where each head uses quantum circuits
    for attention computation.

    Args:
        embed_dim: Embedding dimension
        n_heads: Number of attention heads
        n_qubits: Qubits per head
        dropout: Dropout probability
    """

    def __init__(
        self,
        embed_dim: int,
        n_heads: int = 8,
        n_qubits: int = 8,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.n_heads = n_heads
        self.n_qubits = n_qubits
        self.dropout = dropout

        assert embed_dim % n_heads == 0

        self.heads = nn.ModuleList(
            [
                QuantumAttention(
                    embed_dim=embed_dim,
                    n_qubits=n_qubits,
                    n_heads=1,
                )
                for _ in range(n_heads)
            ]
        )

        self.out_proj = nn.Linear(embed_dim, embed_dim)
        self.dropout_layer = nn.Dropout(dropout)

    def forward(
        self,
        x: Tensor,
        mask: Optional[Tensor] = None,
    ) -> Tuple[Tensor, Optional[Tensor]]:
        """
        Multi-head attention forward pass.

        Args:
            x: Input [batch, seq_len, embed_dim]
            mask: Optional mask
        Returns:
            output: Attention output
            attn_weights: Combined attention weights
        """
        batch_size = seq_len = x.shape[1]

        outputs = []
        all_weights = []

        for head in self.heads:
            out, weights = head(x, x, x, mask)
            outputs.append(out)
            all_weights.append(weights)

        output = torch.stack(outputs, dim=0).sum(dim=0)
        output = self.dropout_layer(output)
        output = self.out_proj(output)

        return output, all_weights[0] if all_weights else None


class QuantumSelfAttention(nn.Module):
    """
    Quantum Self-Attention Layer.

    Self-attention where query, key, and value all come
    from the same source, processed quantum-mechanically.

    Args:
        embed_dim: Embedding dimension
        n_qubits: Number of qubits
        dropout: Dropout probability
    """

    def __init__(
        self,
        embed_dim: int,
        n_qubits: int = 8,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.n_qubits = n_qubits

        self.attention = QuantumAttention(
            embed_dim=embed_dim,
            n_qubits=n_qubits,
            n_heads=1,
        )

        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)

        self.ff = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 4),
            nn.GELU(),
            nn.Linear(embed_dim * 4, embed_dim),
        )

        self.dropout = nn.Dropout(dropout)

    def forward(self, x: Tensor, mask: Optional[Tensor] = None) -> Tensor:
        """
        Self-attention forward pass with residual connections.

        Args:
            x: Input [batch, seq_len, embed_dim]
            mask: Optional mask
        Returns:
            Output [batch, seq_len, embed_dim]
        """
        attn_out, _ = self.attention(x, x, x, mask)
        x = self.norm1(x + self.dropout(attn_out))

        ff_out = self.ff(x)
        x = self.norm2(x + ff_out)

        return x


class QuantumCrossAttention(nn.Module):
    """
    Quantum Cross-Attention.

    Cross-attention where queries come from one sequence
    while keys and values come from another.

    Args:
        embed_dim: Embedding dimension
        n_qubits: Number of qubits
        dropout: Dropout probability
    """

    def __init__(
        self,
        embed_dim: int,
        n_qubits: int = 8,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.n_qubits = n_qubits

        self.attention = QuantumAttention(
            embed_dim=embed_dim,
            n_qubits=n_qubits,
            n_heads=1,
        )

        self.norm = nn.LayerNorm(embed_dim)

    def forward(
        self,
        query: Tensor,
        context: Tensor,
        mask: Optional[Tensor] = None,
    ) -> Tensor:
        """
        Cross-attention forward pass.

        Args:
            query: Query sequence [batch, query_len, embed_dim]
            context: Context sequence [batch, context_len, embed_dim]
            mask: Optional mask
        Returns:
            Output [batch, query_len, embed_dim]
        """
        attn_out, _ = self.attention(query, context, context, mask)
        return self.norm(query + attn_out)


class QuantumTransformerLayer(nn.Module):
    """
    Quantum Transformer Layer.

    Complete transformer encoder layer with quantum
    attention and feed-forward network.

    Args:
        embed_dim: Embedding dimension
        n_qubits: Number of qubits
        n_heads: Number of attention heads
        ff_dim: Feed-forward dimension
        dropout: Dropout probability
    """

    def __init__(
        self,
        embed_dim: int,
        n_qubits: int = 8,
        n_heads: int = 8,
        ff_dim: Optional[int] = None,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.n_qubits = n_qubits

        ff_dim = ff_dim or embed_dim * 4

        self.attention = QuantumMultiHeadAttention(
            embed_dim=embed_dim,
            n_heads=n_heads,
            n_qubits=n_qubits,
            dropout=dropout,
        )

        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)

        self.ff = nn.Sequential(
            nn.Linear(embed_dim, ff_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ff_dim, embed_dim),
            nn.Dropout(dropout),
        )

    def forward(
        self,
        x: Tensor,
        mask: Optional[Tensor] = None,
    ) -> Tensor:
        """
        Transformer layer forward pass.

        Args:
            x: Input [batch, seq_len, embed_dim]
            mask: Optional mask
        Returns:
            Output [batch, seq_len, embed_dim]
        """
        attn_out, _ = self.attention(x, mask)
        x = self.norm1(x + attn_out)

        ff_out = self.ff(x)
        x = self.norm2(x + ff_out)

        return x


class QuantumTransformer(nn.Module):
    """
    Quantum Transformer Model.

    Complete transformer architecture with quantum
    attention mechanisms.

    Args:
        vocab_size: Vocabulary size
        embed_dim: Embedding dimension
        n_layers: Number of transformer layers
        n_heads: Number of attention heads
        n_qubits: Qubits per attention head
        max_seq_len: Maximum sequence length
        dropout: Dropout probability
    """

    def __init__(
        self,
        vocab_size: int,
        embed_dim: int = 512,
        n_layers: int = 6,
        n_heads: int = 8,
        n_qubits: int = 8,
        max_seq_len: int = 512,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.embed_dim = embed_dim

        self.token_embedding = nn.Embedding(vocab_size, embed_dim)
        self.pos_embedding = nn.Embedding(max_seq_len, embed_dim)

        self.layers = nn.ModuleList(
            [
                QuantumTransformerLayer(
                    embed_dim=embed_dim,
                    n_qubits=n_qubits,
                    n_heads=n_heads,
                    dropout=dropout,
                )
                for _ in range(n_layers)
            ]
        )

        self.norm = nn.LayerNorm(embed_dim)
        self.head = nn.Linear(embed_dim, vocab_size)

        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        x: Tensor,
        mask: Optional[Tensor] = None,
    ) -> Tensor:
        """
        Transformer forward pass.

        Args:
            x: Input token indices [batch, seq_len]
            mask: Optional mask
        Returns:
            Logits [batch, seq_len, vocab_size]
        """
        batch_size, seq_len = x.shape

        tok_emb = self.token_embedding(x)

        positions = torch.arange(seq_len, device=x.device)
        pos_emb = self.pos_embedding(positions)

        x = self.dropout(tok_emb + pos_emb)

        for layer in self.layers:
            x = layer(x, mask)

        x = self.norm(x)

        return self.head(x)
