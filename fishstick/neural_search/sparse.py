from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


@dataclass
class BM25Config:
    k1: float = 1.5
    b: float = 0.75
    avg_doc_len: float = 100.0
    epsilon: float = 0.25


@dataclass
class LearnedSparseConfig:
    vocab_size: int = 30522
    embedding_dim: int = 768
    max_seq_length: int = 512
    num_layers: int = 4
    num_heads: int = 8
    dropout: float = 0.1
    normalize: bool = True


@dataclass
class SPLADEConfig:
    vocab_size: int = 30522
    embedding_dim: int = 768
    max_seq_length: int = 512
    num_layers: int = 4
    num_heads: int = 8
    dropout: float = 0.1
    temperature: float = 0.1
    max_doc_tokens: int = 100


class SparseRetriever(nn.Module):
    def __init__(self, vocab_size: int = 30522):
        super().__init__()
        self.vocab_size = vocab_size

    def encode_queries(self, queries: Tensor) -> Tensor:
        raise NotImplementedError

    def encode_documents(self, documents: Tensor) -> Tensor:
        raise NotImplementedError

    def retrieve(
        self,
        queries: Tensor,
        documents: Tensor,
        top_k: int = 10,
    ) -> Tuple[Tensor, Tensor]:
        query_weights = self.encode_queries(queries)
        doc_weights = self.encode_documents(documents)

        scores = torch.matmul(query_weights, doc_weights.T)
        top_scores, top_indices = torch.topk(
            scores, k=min(top_k, scores.size(1)), dim=1
        )
        return top_indices, top_scores


class BM25:
    def __init__(self, config: Optional[BM25Config] = None):
        self.config = config or BM25Config()
        self.doc_freqs: Dict[int, int] = {}
        self.avg_doc_len: float = self.config.avg_doc_len
        self.doc_count: int = 0
        self.idf: Dict[int, float] = {}

    def fit(self, corpus: List[List[int]]) -> None:
        self.doc_count = len(corpus)
        doc_lens = []
        df = {}

        for doc in corpus:
            doc_lens.append(len(doc))
            seen = set()
            for token in doc:
                if token not in seen:
                    df[token] = df.get(token, 0) + 1
                    seen.add(token)

        self.doc_freqs = df
        self.avg_doc_len = sum(doc_lens) / len(doc_lens) if doc_lens else 1.0

        for token, freq in df.items():
            self.idf[token] = torch.log(
                (self.doc_count - freq + 0.5) / (freq + 0.5) + 1
            ).item()

    def score(self, query: List[int], document: List[int]) -> float:
        doc_len = len(document)
        doc_freqs = {}
        for token in document:
            doc_freqs[token] = doc_freqs.get(token, 0) + 1

        score = 0.0
        for token in query:
            if token in doc_freqs:
                freq = doc_freqs[token]
                idf = self.idf.get(token, 0.0)
                numerator = freq * (self.config.k1 + 1)
                denominator = freq + self.config.k1 * (
                    1 - self.config.b + self.config.b * doc_len / self.avg_doc_len
                )
                score += idf * numerator / denominator

        return score

    def get_scores(self, query: List[int], documents: List[List[int]]) -> List[float]:
        return [self.score(query, doc) for doc in documents]

    def get_top_k(
        self, query: List[int], documents: List[List[int]], k: int = 10
    ) -> Tuple[List[int], List[float]]:
        scores = self.get_scores(query, documents)
        top_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[
            :k
        ]
        top_scores = [scores[i] for i in top_indices]
        return top_indices, top_scores


class LearnedSparseRetrieval(SparseRetriever):
    def __init__(self, config: Optional[LearnedSparseConfig] = None):
        config = config or LearnedSparseConfig()
        super().__init__(vocab_size=config.vocab_size)
        self.config = config

        self.embedding = nn.Embedding(
            config.vocab_size, config.embedding_dim, padding_idx=0
        )
        self.encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=config.embedding_dim,
                nhead=config.num_heads,
                dim_feedforward=config.embedding_dim * 4,
                dropout=config.dropout,
                activation="gelu",
                batch_first=True,
            ),
            num_layers=config.num_layers,
        )

        self.token_scorer = nn.Linear(config.embedding_dim, 1)
        self.normalize = config.normalize

    def encode_queries(self, queries: Tensor) -> Tensor:
        x = self.embedding(queries)
        x = self.encoder(x)
        weights = self.token_scorer(x).squeeze(-1)
        weights = F.relu(weights)
        if self.normalize:
            weights = F.normalize(weights, p=1, dim=-1)
        return weights

    def encode_documents(self, documents: Tensor) -> Tensor:
        x = self.embedding(documents)
        x = self.encoder(x)
        weights = self.token_scorer(x).squeeze(-1)
        weights = F.relu(weights)
        if self.normalize:
            weights = F.normalize(weights, p=1, dim=-1)
        return weights

    def forward(
        self,
        queries: Optional[Tensor] = None,
        documents: Optional[Tensor] = None,
    ) -> Union[Tensor, Tuple[Tensor, Tensor]]:
        if queries is not None and documents is not None:
            query_weights = self.encode_queries(queries)
            doc_weights = self.encode_documents(documents)
            return query_weights, doc_weights
        elif queries is not None:
            return self.encode_queries(queries)
        elif documents is not None:
            return self.encode_documents(documents)
        else:
            raise ValueError("Either queries or documents must be provided")


class SPLADE(SparseRetriever):
    def __init__(self, config: Optional[SPLADEConfig] = None):
        config = config or SPLADEConfig()
        super().__init__(vocab_size=config.vocab_size)
        self.config = config

        self.embedding = nn.Embedding(
            config.vocab_size, config.embedding_dim, padding_idx=0
        )
        self.position_embedding = nn.Embedding(
            config.max_seq_length, config.embedding_dim
        )

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=config.embedding_dim,
            nhead=config.num_heads,
            dim_feedforward=config.embedding_dim * 4,
            dropout=config.dropout,
            activation="gelu",
            batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(
            encoder_layer, num_layers=config.num_layers
        )

        self.term_importance = nn.Linear(config.embedding_dim, 1)
        self.temperature = config.temperature

    def encode_queries(self, queries: Tensor) -> Tensor:
        batch_size, seq_len = queries.shape

        positions = (
            torch.arange(seq_len, device=queries.device)
            .unsqueeze(0)
            .expand(batch_size, -1)
        )
        x = self.embedding(queries) + self.position_embedding(positions)
        x = self.encoder(x)

        term_weights = self.term_importance(x).squeeze(-1)
        sparse_weights = F.log_softmax(term_weights / self.temperature, dim=-1)
        sparse_weights = torch.exp(sparse_weights)

        return sparse_weights

    def encode_documents(self, documents: Tensor) -> Tensor:
        batch_size, seq_len = documents.shape

        positions = (
            torch.arange(seq_len, device=documents.device)
            .unsqueeze(0)
            .expand(batch_size, -1)
        )
        x = self.embedding(documents) + self.position_embedding(positions)
        x = self.encoder(x)

        term_weights = self.term_importance(x).squeeze(-1)
        sparse_weights = F.log_softmax(term_weights / self.temperature, dim=-1)
        sparse_weights = torch.exp(sparse_weights)

        return sparse_weights

    def compute_sparse_weights(self, queries: Tensor) -> Tensor:
        return self.encode_queries(queries)

    def flops_pooling(self, sparse_weights: Tensor, top_k: int = 100) -> Tensor:
        max_vals, max_indices = torch.topk(
            sparse_weights, k=min(top_k, sparse_weights.size(-1)), dim=-1
        )
        pooled = torch.zeros_like(sparse_weights)
        pooled.scatter_(-1, max_indices, max_vals)
        return pooled

    def forward(
        self,
        queries: Optional[Tensor] = None,
        documents: Optional[Tensor] = None,
        pool_top_k: bool = True,
    ) -> Union[Tensor, Tuple[Tensor, Tensor]]:
        if queries is not None:
            query_weights = self.encode_queries(queries)
            if pool_top_k:
                query_weights = self.flops_pooling(
                    query_weights, self.config.max_doc_tokens
                )
            if documents is not None:
                doc_weights = self.encode_documents(documents)
                if pool_top_k:
                    doc_weights = self.flops_pooling(
                        doc_weights, self.config.max_doc_tokens
                    )
                return query_weights, doc_weights
            return query_weights
        elif documents is not None:
            doc_weights = self.encode_documents(documents)
            if pool_top_k:
                doc_weights = self.flops_pooling(
                    doc_weights, self.config.max_doc_tokens
                )
            return doc_weights
        else:
            raise ValueError("Either queries or documents must be provided")


class TFIDF:
    def __init__(self, vocab_size: int = 30522):
        self.vocab_size = vocab_size
        self.doc_freqs: Dict[int, int] = {}
        self.doc_count: int = 0

    def fit(self, corpus: List[List[int]]) -> None:
        self.doc_count = len(corpus)
        df = {}

        for doc in corpus:
            seen = set()
            for token in doc:
                if token not in seen:
                    df[token] = df.get(token, 0) + 1
                    seen.add(token)

        self.doc_freqs = df

    def transform(self, documents: List[List[int]]) -> Tensor:
        n_docs = len(documents)
        tfidf_matrix = torch.zeros(n_docs, self.vocab_size)

        for i, doc in enumerate(documents):
            doc_len = len(doc) if doc else 1
            tf = {}
            for token in doc:
                tf[token] = tf.get(token, 0) + 1

            for token, freq in tf.items():
                tf[token] = freq / doc_len

            for token, tf_val in tf.items():
                df = self.doc_freqs.get(token, 0)
                if df > 0:
                    idf = torch.log(torch.tensor(self.doc_count / df))
                    tfidf_matrix[i, token] = tf_val * idf

        return tfidf_matrix

    def fit_transform(self, corpus: List[List[int]]) -> Tensor:
        self.fit(corpus)
        return self.transform(corpus)
