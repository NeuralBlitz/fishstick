from dataclasses import dataclass
from typing import List, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


@dataclass
class DPRConfig:
    query_encoder_dim: int = 768
    context_encoder_dim: int = 768
    hidden_dim: int = 768
    num_layers: int = 1
    num_heads: int = 8
    dropout: float = 0.1
    max_seq_length: int = 512
    normalize: bool = True


@dataclass
class BiEncoderConfig:
    embedding_dim: int = 768
    hidden_dim: int = 768
    num_layers: int = 6
    num_heads: int = 8
    dropout: float = 0.1
    max_seq_length: int = 512
    normalize: bool = True
    pooling: str = "cls"


@dataclass
class CrossEncoderConfig:
    hidden_dim: int = 768
    num_layers: int = 6
    num_heads: int = 8
    dropout: float = 0.1
    max_seq_length: int = 512
    num_labels: int = 1


@dataclass
class ColBERTConfig:
    embedding_dim: int = 128
    hidden_dim: int = 768
    num_layers: int = 6
    num_heads: int = 8
    dropout: float = 0.1
    max_seq_length: int = 512
    similarity_metric: str = "cosine"
    mask_punctuation: bool = True


class DenseRetriever(nn.Module):
    def __init__(
        self,
        embedding_dim: int = 768,
        normalize: bool = True,
    ):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.normalize = normalize

    def encode_queries(self, queries: Tensor) -> Tensor:
        raise NotImplementedError

    def encode_documents(self, documents: Tensor) -> Tensor:
        raise NotImplementedError

    def compute_scores(self, query_embeds: Tensor, doc_embeds: Tensor) -> Tensor:
        if self.normalize:
            query_embeds = F.normalize(query_embeds, p=2, dim=-1)
            doc_embeds = F.normalize(doc_embeds, p=2, dim=-1)
        return torch.matmul(query_embeds, doc_embeds.T)

    def retrieve(
        self,
        queries: Tensor,
        documents: Tensor,
        top_k: int = 10,
    ) -> Tuple[Tensor, Tensor]:
        query_embeds = self.encode_queries(queries)
        doc_embeds = self.encode_documents(documents)
        scores = self.compute_scores(query_embeds, doc_embeds)
        top_scores, top_indices = torch.topk(
            scores, k=min(top_k, scores.size(1)), dim=1
        )
        return top_indices, top_scores


class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 512, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2) * (-torch.log(torch.tensor(10000.0)) / d_model)
        )
        pe = torch.zeros(max_len, d_model)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe)

    def forward(self, x: Tensor) -> Tensor:
        x = x + self.pe[: x.size(0)]
        return self.dropout(x)


class TransformerEncoder(nn.Module):
    def __init__(
        self,
        embedding_dim: int,
        num_layers: int,
        num_heads: int,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.layers = nn.ModuleList(
            [
                nn.TransformerEncoderLayer(
                    d_model=embedding_dim,
                    nhead=num_heads,
                    dim_feedforward=embedding_dim * 4,
                    dropout=dropout,
                    activation="gelu",
                    batch_first=True,
                )
                for _ in range(num_layers)
            ]
        )

    def forward(self, x: Tensor, mask: Optional[Tensor] = None) -> Tensor:
        for layer in self.layers:
            x = layer(x, src_key_padding_mask=mask)
        return x


class DPR(DenseRetriever):
    def __init__(self, vocab_size: int = 30522, config: Optional[DPRConfig] = None):
        config = config or DPRConfig()
        super().__init__(
            embedding_dim=config.query_encoder_dim, normalize=config.normalize
        )
        self.config = config

        self.query_encoder = nn.TransformerEncoder(
            d_model=config.query_encoder_dim,
            nhead=config.num_heads,
            dim_feedforward=config.hidden_dim * 4,
            dropout=config.dropout,
            activation="gelu",
            num_layers=config.num_layers,
            batch_first=True,
        )
        self.context_encoder = nn.TransformerEncoder(
            d_model=config.context_encoder_dim,
            nhead=config.num_heads,
            dim_feedforward=config.hidden_dim * 4,
            dropout=config.dropout,
            activation="gelu",
            num_layers=config.num_layers,
            batch_first=True,
        )

        self.query_projection = nn.Linear(
            config.query_encoder_dim, config.query_encoder_dim
        )
        self.context_projection = nn.Linear(
            config.context_encoder_dim, config.context_encoder_dim
        )
        self.position_encoding = PositionalEncoding(
            config.query_encoder_dim, config.max_seq_length, config.dropout
        )

    def encode_queries(self, queries: Tensor) -> Tensor:
        x = self.position_encoding(queries)
        x = self.query_encoder(x)
        x = x.mean(dim=1)
        return self.query_projection(x)

    def encode_documents(self, documents: Tensor) -> Tensor:
        x = self.position_encoding(documents)
        x = self.context_encoder(x)
        x = x.mean(dim=1)
        return self.context_projection(x)

    def compute_context_representations(self, documents: Tensor) -> Tensor:
        return self.encode_documents(documents)

    def compute_query_representations(self, queries: Tensor) -> Tensor:
        return self.encode_queries(queries)


class BiEncoder(DenseRetriever):
    def __init__(
        self, vocab_size: int = 30522, config: Optional[BiEncoderConfig] = None
    ):
        config = config or BiEncoderConfig()
        super().__init__(embedding_dim=config.embedding_dim, normalize=config.normalize)
        self.config = config

        self.embedding = nn.Embedding(vocab_size, config.embedding_dim, padding_idx=0)
        self.position_encoding = PositionalEncoding(
            config.embedding_dim, config.max_seq_length, config.dropout
        )

        self.encoder = TransformerEncoder(
            embedding_dim=config.embedding_dim,
            num_layers=config.num_layers,
            num_heads=config.num_heads,
            dropout=config.dropout,
        )

        self.pooling = config.pooling

    def encode_queries(self, queries: Tensor) -> Tensor:
        x = self.embedding(queries)
        x = self.position_encoding(x)
        x = self.encoder(x)
        return self._pool(x, queries.ne(0))

    def encode_documents(self, documents: Tensor) -> Tensor:
        x = self.embedding(documents)
        x = self.position_encoding(x)
        x = self.encoder(x)
        return self._pool(x, documents.ne(0))

    def _pool(self, x: Tensor, mask: Tensor) -> Tensor:
        if self.pooling == "cls":
            return x[:, 0]
        elif self.pooling == "mean":
            mask_expanded = mask.unsqueeze(-1).float()
            return (x * mask_expanded).sum(dim=1) / mask_expanded.sum(dim=1).clamp(
                min=1e-9
            )
        elif self.pooling == "max":
            mask_expanded = mask.unsqueeze(-1).float()
            x = x.masked_fill(~mask, float("-inf"))
            return x.max(dim=1).values
        else:
            raise ValueError(f"Unknown pooling: {self.pooling}")

    def forward(
        self,
        queries: Optional[Tensor] = None,
        documents: Optional[Tensor] = None,
    ) -> Union[Tensor, Tuple[Tensor, Tensor]]:
        if queries is not None and documents is not None:
            query_embeds = self.encode_queries(queries)
            doc_embeds = self.encode_documents(documents)
            return query_embeds, doc_embeds
        elif queries is not None:
            return self.encode_queries(queries)
        elif documents is not None:
            return self.encode_documents(documents)
        else:
            raise ValueError("Either queries or documents must be provided")


class CrossEncoder(nn.Module):
    def __init__(
        self, vocab_size: int = 30522, config: Optional[CrossEncoderConfig] = None
    ):
        super().__init__()
        config = config or CrossEncoderConfig()
        self.config = config

        self.embedding = nn.Embedding(vocab_size, config.hidden_dim, padding_idx=0)
        self.position_encoding = PositionalEncoding(
            config.hidden_dim, config.max_seq_length, config.dropout
        )

        self.encoder = TransformerEncoder(
            embedding_dim=config.hidden_dim,
            num_layers=config.num_layers,
            num_heads=config.num_heads,
            dropout=config.dropout,
        )

        self.classifier = nn.Sequential(
            nn.Linear(config.hidden_dim, config.hidden_dim),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.hidden_dim, config.num_labels),
        )

    def forward(self, query_doc_pairs: Tensor) -> Tensor:
        x = self.embedding(query_doc_pairs)
        x = self.position_encoding(x)
        x = self.encoder(x)
        x = x.mean(dim=1)
        return self.classifier(x)

    def score(self, queries: Tensor, documents: Tensor) -> Tensor:
        sep_token_id = 2
        batch_size = queries.size(0)

        sep_mask = (documents == sep_token_id).cumsum(dim=1) == 0
        combined = torch.cat([queries, documents], dim=1)

        return self.forward(combined)


class ColBERT(DenseRetriever):
    def __init__(self, vocab_size: int = 30522, config: Optional[ColBERTConfig] = None):
        config = config or ColBERTConfig()
        super().__init__(embedding_dim=config.embedding_dim, normalize=config.normalize)
        self.config = config

        self.embedding = nn.Embedding(vocab_size, config.hidden_dim, padding_idx=0)
        self.position_encoding = PositionalEncoding(
            config.hidden_dim, config.max_seq_length, config.dropout
        )

        self.encoder = TransformerEncoder(
            embedding_dim=config.hidden_dim,
            num_layers=config.num_layers,
            num_heads=config.num_heads,
            dropout=config.dropout,
        )

        self.linear = nn.Linear(config.hidden_dim, config.embedding_dim)

        self.similarity_metric = config.similarity_metric

    def encode_queries(self, queries: Tensor) -> Tensor:
        x = self.embedding(queries)
        x = self.position_encoding(x)
        x = self.encoder(x)
        x = self.linear(x)
        if self.normalize:
            x = F.normalize(x, p=2, dim=-1)
        return x

    def encode_documents(self, documents: Tensor) -> Tensor:
        x = self.embedding(documents)
        x = self.position_encoding(x)
        x = self.encoder(x)
        x = self.linear(x)
        if self.normalize:
            x = F.normalize(x, p=2, dim=-1)
        return x

    def compute_scores(self, query_embeds: Tensor, doc_embeds: Tensor) -> Tensor:
        query_embeds = query_embeds.unsqueeze(2)
        doc_embeds = doc_embeds.unsqueeze(1)

        if self.similarity_metric == "cosine":
            scores = (query_embeds * doc_embeds).sum(dim=-1)
        elif self.similarity_metric == "maxsim":
            scores = (query_embeds * doc_embeds).sum(dim=-1)
            scores = scores.max(dim=-1).values.sum(dim=-1)
        else:
            raise ValueError(f"Unknown similarity metric: {self.similarity_metric}")

        return scores

    def max_sim_score(self, query_embeds: Tensor, doc_embeds: Tensor) -> Tensor:
        scores = torch.matmul(query_embeds, doc_embeds.transpose(1, 2))
        max_scores = scores.max(dim=2).values
        return max_scores.sum(dim=1)
