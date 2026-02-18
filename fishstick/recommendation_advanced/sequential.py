import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, List
import math


class MultiHeadAttention(nn.Module):
    def __init__(self, embed_dim: int, num_heads: int, dropout: float = 0.1):
        super().__init__()
        assert embed_dim % num_heads == 0

        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads

        self.q_linear = nn.Linear(embed_dim, embed_dim)
        self.k_linear = nn.Linear(embed_dim, embed_dim)
        self.v_linear = nn.Linear(embed_dim, embed_dim)
        self.out_linear = nn.Linear(embed_dim, embed_dim)

        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        batch_size = query.size(0)

        Q = (
            self.q_linear(query)
            .view(batch_size, -1, self.num_heads, self.head_dim)
            .transpose(1, 2)
        )
        K = (
            self.k_linear(key)
            .view(batch_size, -1, self.num_heads, self.head_dim)
            .transpose(1, 2)
        )
        V = (
            self.v_linear(value)
            .view(batch_size, -1, self.num_heads, self.head_dim)
            .transpose(1, 2)
        )

        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.head_dim)

        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)

        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)

        context = torch.matmul(attn_weights, V)
        context = (
            context.transpose(1, 2).contiguous().view(batch_size, -1, self.embed_dim)
        )

        output = self.out_linear(context)
        return output, attn_weights


class FeedForward(nn.Module):
    def __init__(self, embed_dim: int, ff_dim: int = 2048, dropout: float = 0.1):
        super().__init__()
        self.linear1 = nn.Linear(embed_dim, ff_dim)
        self.linear2 = nn.Linear(ff_dim, embed_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.linear1(x))
        x = self.dropout(x)
        x = self.linear2(x)
        return x


class TransformerBlock(nn.Module):
    def __init__(
        self, embed_dim: int, num_heads: int, ff_dim: int = 2048, dropout: float = 0.1
    ):
        super().__init__()
        self.attention = MultiHeadAttention(embed_dim, num_heads, dropout)
        self.feed_forward = FeedForward(embed_dim, ff_dim, dropout)
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self, x: torch.Tensor, mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        attn_output, _ = self.attention(x, x, x, mask)
        x = self.norm1(x + self.dropout(attn_output))

        ff_output = self.feed_forward(x)
        x = self.norm2(x + self.dropout(ff_output))

        return x


class SASRec(nn.Module):
    def __init__(
        self,
        num_items: int,
        embedding_dim: int = 128,
        num_heads: int = 2,
        num_layers: 2 = 2,
        ff_dim: int = 512,
        dropout: float = 0.1,
        max_seq_length: int = 50,
    ):
        super().__init__()
        self.num_items = num_items
        self.embedding_dim = embedding_dim
        self.max_seq_length = max_seq_length

        self.item_embedding = nn.Embedding(num_items + 1, embedding_dim, padding_idx=0)
        self.position_embedding = nn.Embedding(max_seq_length, embedding_dim)

        self.transformer_blocks = nn.ModuleList(
            [
                TransformerBlock(embedding_dim, num_heads, ff_dim, dropout)
                for _ in range(num_layers)
            ]
        )

        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(embedding_dim)

        self._init_weights()

    def _init_weights(self):
        nn.init.normal_(self.item_embedding.weight, std=0.02)
        nn.init.normal_(self.position_embedding.weight, std=0.02)

    def forward(
        self,
        seq_items: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        batch_size, seq_len = seq_items.size()

        item_embeddings = self.item_embedding(seq_items)

        positions = (
            torch.arange(seq_len, device=seq_items.device)
            .unsqueeze(0)
            .expand(batch_size, -1)
        )
        position_embeddings = self.position_embedding(positions)

        x = item_embeddings + position_embeddings
        x = self.dropout(x)
        x = self.layer_norm(x)

        causal_mask = torch.triu(
            torch.ones(seq_len, seq_len, device=seq_items.device), diagonal=1
        ).bool()

        for block in self.transformer_blocks:
            x = block(x, causal_mask)

        return x

    def predict(self, seq_items: torch.Tensor) -> torch.Tensor:
        x = self.forward(seq_items)
        item_embeddings = self.item_embedding.weight[:-1]
        scores = torch.matmul(x, item_embeddings.transpose(0, 1))
        return scores


class BERT4Rec(nn.Module):
    def __init__(
        self,
        num_items: int,
        embedding_dim: int = 128,
        num_heads: int = 2,
        num_layers: int = 2,
        ff_dim: int = 512,
        dropout: float = 0.1,
        max_seq_length: int = 50,
    ):
        super().__init__()
        self.num_items = num_items
        self.embedding_dim = embedding_dim
        self.max_seq_length = max_seq_length

        self.item_embedding = nn.Embedding(num_items + 1, embedding_dim, padding_idx=0)
        self.position_embedding = nn.Embedding(max_seq_length, embedding_dim)

        self.transformer_blocks = nn.ModuleList(
            [
                TransformerBlock(embedding_dim, num_heads, ff_dim, dropout)
                for _ in range(num_layers)
            ]
        )

        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(embedding_dim)

        self._init_weights()

    def _init_weights(self):
        nn.init.normal_(self.item_embedding.weight, std=0.02)
        nn.init.normal_(self.position_embedding.weight, std=0.02)

    def forward(
        self,
        seq_items: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        batch_size, seq_len = seq_items.size()

        item_embeddings = self.item_embedding(seq_items)

        positions = (
            torch.arange(seq_len, device=seq_items.device)
            .unsqueeze(0)
            .expand(batch_size, -1)
        )
        position_embeddings = self.position_embedding(positions)

        x = item_embeddings + position_embeddings
        x = self.dropout(x)
        x = self.layer_norm(x)

        for block in self.transformer_blocks:
            x = block(x)

        return x

    def predict(self, seq_items: torch.Tensor) -> torch.Tensor:
        x = self.forward(seq_items)
        item_embeddings = self.item_embedding.weight[:-1]
        scores = torch.matmul(x, item_embeddings.transpose(0, 1))
        return scores


class GRU4Rec(nn.Module):
    def __init__(
        self,
        num_items: int,
        embedding_dim: int = 128,
        hidden_dim: int = 256,
        num_layers: int = 2,
        dropout: float = 0.2,
    ):
        super().__init__()
        self.num_items = num_items
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

        self.item_embedding = nn.Embedding(num_items + 1, embedding_dim, padding_idx=0)

        self.gru = nn.GRU(
            embedding_dim,
            hidden_dim,
            num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
        )

        self.dropout = nn.Dropout(dropout)
        self.output_layer = nn.Linear(hidden_dim, num_items)

        self._init_weights()

    def _init_weights(self):
        nn.init.normal_(self.item_embedding.weight, std=0.01)

    def forward(
        self,
        seq_items: torch.Tensor,
        hidden: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        embedded = self.item_embedding(seq_items)
        embedded = self.dropout(embedded)

        if hidden is None:
            hidden = torch.zeros(
                self.num_layers,
                embedded.size(0),
                self.hidden_dim,
                device=seq_items.device,
            )

        output, hidden = self.gru(embedded, hidden)
        output = self.dropout(output)

        scores = self.output_layer(output)
        return scores, hidden

    def predict(self, seq_items: torch.Tensor) -> torch.Tensor:
        scores, _ = self.forward(seq_items)
        return scores


class TimeAwareRec(nn.Module):
    def __init__(
        self,
        num_items: int,
        embedding_dim: int = 128,
        hidden_dim: int = 256,
        num_layers: int = 2,
        dropout: float = 0.2,
        time_embedding_dim: int = 32,
    ):
        super().__init__()
        self.num_items = num_items
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

        self.item_embedding = nn.Embedding(num_items + 1, embedding_dim, padding_idx=0)
        self.time_embedding = nn.Embedding(1000, time_embedding_dim)

        combined_dim = embedding_dim + time_embedding_dim

        self.gru = nn.GRU(
            combined_dim,
            hidden_dim,
            num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
        )

        self.dropout = nn.Dropout(dropout)
        self.output_layer = nn.Linear(hidden_dim, num_items)

        self.time_attention = nn.Linear(time_embedding_dim, 1)

        self._init_weights()

    def _init_weights(self):
        nn.init.normal_(self.item_embedding.weight, std=0.01)
        nn.init.normal_(self.time_embedding.weight, std=0.01)

    def forward(
        self,
        seq_items: torch.Tensor,
        seq_times: torch.Tensor,
        hidden: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        item_emb = self.item_embedding(seq_items)

        time_emb = self.time_embedding(torch.clamp(seq_times, 0, 999))

        time_weights = F.softmax(self.time_attention(time_emb), dim=1)
        time_emb_weighted = time_emb * time_weights

        combined = torch.cat([item_emb, time_emb_weighted], dim=-1)
        combined = self.dropout(combined)

        if hidden is None:
            hidden = torch.zeros(
                self.num_layers,
                combined.size(0),
                self.hidden_dim,
                device=seq_items.device,
            )

        output, hidden = self.gru(combined, hidden)
        output = self.dropout(output)

        scores = self.output_layer(output)
        return scores, hidden

    def predict(self, seq_items: torch.Tensor, seq_times: torch.Tensor) -> torch.Tensor:
        scores, _ = self.forward(seq_items, seq_times)
        return scores


class SessionRNN(nn.Module):
    def __init__(
        self,
        num_items: int,
        embedding_dim: int = 128,
        hidden_dim: int = 256,
        num_layers: int = 2,
        dropout: float = 0.2,
    ):
        super().__init__()
        self.num_items = num_items
        self.hidden_dim = hidden_dim

        self.item_embedding = nn.Embedding(num_items + 1, embedding_dim, padding_idx=0)

        self.gru = nn.GRU(
            embedding_dim,
            hidden_dim,
            num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
        )

        self.dropout = nn.Dropout(dropout)
        self.output_layer = nn.Linear(hidden_dim, num_items)

    def forward(
        self,
        seq_items: torch.Tensor,
        hidden: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        embedded = self.item_embedding(seq_items)
        embedded = self.dropout(embedded)

        if hidden is None:
            hidden = torch.zeros(
                self.gru.num_layers,
                embedded.size(0),
                self.hidden_dim,
                device=seq_items.device,
            )

        output, hidden = self.gru(embedded, hidden)
        output = self.dropout(output)

        last_output = output[:, -1, :]
        scores = self.output_layer(last_output)

        return scores, hidden
