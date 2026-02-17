"""
NLP Embeddings
"""

import torch
from torch import nn, Tensor
import math


class WordEmbedding(nn.Module):
    """Word embedding layer."""

    def __init__(
        self,
        vocab_size: int,
        embed_dim: int,
        padding_idx: int = 0,
    ):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=padding_idx)
        self.embed_dim = embed_dim

        # Initialize
        self._init_weights()

    def _init_weights(self):
        """Initialize weights."""
        nn.init.normal_(self.embedding.weight, mean=0, std=0.02)

    def forward(self, x: Tensor) -> Tensor:
        return self.embedding(x) * math.sqrt(self.embed_dim)


class PositionalEncoding(nn.Module):
    """Sinusoidal positional encoding."""

    def __init__(
        self,
        d_model: int,
        max_len: int = 5000,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        # Create positional encoding
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
        x = x + self.pe[:, : x.size(1), :]
        return self.dropout(x)


class RotaryPositionalEmbedding(nn.Module):
    """Rotary positional embedding (RoPE)."""

    def __init__(self, dim: int, max_seq_len: int = 2048):
        super().__init__()
        inv_freq = 1.0 / (10000 ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq)

    def forward(self, x: Tensor, seq_len: int) -> Tensor:
        t = torch.arange(seq_len, device=x.device).type_as(self.inv_freq)
        freqs = torch.einsum("i,j->ij", t, self.inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)
        return emb[None, :, :]


class CharacterEmbedding(nn.Module):
    """Character-level embedding."""

    def __init__(
        self,
        vocab_size: int,
        embed_dim: int,
        kernel_sizes: list = [3, 5, 7],
    ):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)

        # Character CNN
        self.convs = nn.ModuleList(
            [nn.Conv1d(embed_dim, embed_dim, k, padding=k // 2) for k in kernel_sizes]
        )

    def forward(self, x: Tensor) -> Tensor:
        # x: [batch, seq_len, char_len]
        batch, seq_len, char_len = x.shape
        x = x.view(-1, char_len)  # [batch*seq_len, char_len]

        x = self.embedding(x)  # [batch*seq_len, char_len, embed_dim]
        x = x.permute(0, 2, 1)  # [batch*seq_len, embed_dim, char_len]

        # Apply convolutions
        conv_outputs = []
        for conv in self.convs:
            conv_out = torch.relu(conv(x))
            conv_outputs.append(conv_out.max(dim=-1)[0])

        x = torch.cat(conv_outputs, dim=-1)
        x = x.view(batch, seq_len, -1)

        return x


class SentenceEmbedding(nn.Module):
    """Sentence-level embedding (e.g., mean pooling)."""

    def __init__(self, method: str = "mean"):
        super().__init__()
        self.method = method

    def forward(self, embeddings: Tensor, mask: Tensor = None) -> Tensor:
        """
        Args:
            embeddings: [batch, seq_len, embed_dim]
            mask: [batch, seq_len]
        """
        if mask is None:
            if self.method == "mean":
                return embeddings.mean(dim=1)
            elif self.method == "max":
                return embeddings.max(dim=1)[0]
            elif self.method == "sum":
                return embeddings.sum(dim=1)
        else:
            mask = mask.unsqueeze(-1).float()
            masked_embeddings = embeddings * mask

            if self.method == "mean":
                return masked_embeddings.sum(dim=1) / mask.sum(dim=1).clamp(min=1)
            elif self.method == "max":
                masked_embeddings = masked_embeddings.masked_fill(
                    mask == 0, float("-inf")
                )
                return masked_embeddings.max(dim=1)[0]
            elif self.method == "sum":
                return masked_embeddings.sum(dim=1)

        return embeddings.mean(dim=1)
