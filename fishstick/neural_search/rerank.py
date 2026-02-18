from dataclasses import dataclass
from typing import List, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


@dataclass
class CrossEncoderRerankerConfig:
    hidden_dim: int = 768
    num_layers: int = 6
    num_heads: int = 8
    dropout: float = 0.1
    max_seq_length: int = 512
    num_labels: int = 1


class CrossEncoderReranker(nn.Module):
    def __init__(
        self,
        vocab_size: int = 30522,
        config: Optional[CrossEncoderRerankerConfig] = None,
    ):
        super().__init__()
        config = config or CrossEncoderRerankerConfig()
        self.config = config

        self.embedding = nn.Embedding(vocab_size, config.hidden_dim, padding_idx=0)

        self.encoder_layers = nn.ModuleList(
            [
                nn.TransformerEncoderLayer(
                    d_model=config.hidden_dim,
                    nhead=config.num_heads,
                    dim_feedforward=config.hidden_dim * 4,
                    dropout=config.dropout,
                    activation="gelu",
                    batch_first=True,
                )
                for _ in range(config.num_layers)
            ]
        )

        self.classifier = nn.Sequential(
            nn.Linear(config.hidden_dim, config.hidden_dim),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.hidden_dim, config.num_labels),
        )

    def forward(self, query_doc_pairs: Tensor) -> Tensor:
        x = self.embedding(query_doc_pairs)

        for layer in self.encoder_layers:
            x = layer(x)

        x = x.mean(dim=1)
        return self.classifier(x)

    def rerank(
        self,
        queries: Tensor,
        documents: Tensor,
        top_k: int = 10,
    ) -> Tuple[Tensor, Tensor]:
        batch_size = queries.size(0)
        num_docs = documents.size(1)

        scores_list = []
        for i in range(batch_size):
            query = queries[i].unsqueeze(0).expand(num_docs, -1)
            pairs = torch.cat([query, documents[:, : documents.size(2)]], dim=1)
            scores = self.forward(pairs)
            scores_list.append(scores.squeeze(-1))

        scores = torch.stack(scores_list)
        top_scores, top_indices = torch.topk(scores, k=min(top_k, num_docs), dim=1)

        return top_indices, top_scores


class PointwiseReranker(nn.Module):
    def __init__(
        self,
        vocab_size: int = 30522,
        hidden_dim: int = 768,
        num_layers: int = 6,
        num_heads: int = 8,
        dropout: float = 0.1,
    ):
        super().__init__()

        self.embedding = nn.Embedding(vocab_size, hidden_dim, padding_idx=0)

        self.encoder_layers = nn.ModuleList(
            [
                nn.TransformerEncoderLayer(
                    d_model=hidden_dim,
                    nhead=num_heads,
                    dim_feedforward=hidden_dim * 4,
                    dropout=dropout,
                    activation="gelu",
                    batch_first=True,
                )
                for _ in range(num_layers)
            ]
        )

        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, query_doc_pairs: Tensor) -> Tensor:
        x = self.embedding(query_doc_pairs)

        for layer in self.encoder_layers:
            x = layer(x)

        x = x.mean(dim=1)
        return self.classifier(x)

    def compute_loss(
        self,
        query_doc_pairs: Tensor,
        labels: Tensor,
    ) -> Tensor:
        logits = self.forward(query_doc_pairs)
        return F.mse_loss(logits.squeeze(-1), labels.float())

    def rerank(
        self,
        queries: Tensor,
        documents: Tensor,
        top_k: int = 10,
    ) -> Tuple[Tensor, Tensor]:
        batch_size = queries.size(0)
        num_docs = documents.size(1)

        scores_list = []
        for i in range(batch_size):
            query = queries[i].unsqueeze(0).expand(num_docs, -1)
            pairs = torch.cat([query, documents[:, : documents.size(2)]], dim=1)
            scores = self.forward(pairs)
            scores_list.append(scores.squeeze(-1))

        scores = torch.stack(scores_list)
        top_scores, top_indices = torch.topk(scores, k=min(top_k, num_docs), dim=1)

        return top_indices, top_scores


class PairwiseReranker(nn.Module):
    def __init__(
        self,
        vocab_size: int = 30522,
        hidden_dim: int = 768,
        num_layers: int = 6,
        num_heads: int = 8,
        dropout: float = 0.1,
    ):
        super().__init__()

        self.embedding = nn.Embedding(vocab_size, hidden_dim, padding_idx=0)

        self.encoder_layers = nn.ModuleList(
            [
                nn.TransformerEncoderLayer(
                    d_model=hidden_dim,
                    nhead=num_heads,
                    dim_feedforward=hidden_dim * 4,
                    dropout=dropout,
                    activation="gelu",
                    batch_first=True,
                )
                for _ in range(num_layers)
            ]
        )

        self.scorer = nn.Sequential(
            nn.Linear(hidden_dim * 3, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, query: Tensor, doc_a: Tensor, doc_b: Tensor) -> Tensor:
        x_a = self.embedding(doc_a)
        x_b = self.embedding(doc_b)

        for layer in self.encoder_layers:
            x_a = layer(x_a)
            x_b = layer(x_b)

        x_a = x_a.mean(dim=1)
        x_b = x_b.mean(dim=1)

        combined = torch.cat([x_a, x_b, x_a - x_b], dim=-1)
        return self.scorer(combined)

    def compute_loss(
        self,
        queries: Tensor,
        docs_a: Tensor,
        docs_b: Tensor,
        labels: Tensor,
    ) -> Tensor:
        logits = self.forward(queries, docs_a, docs_b).squeeze(-1)
        return F.binary_cross_entropy_with_logits(logits, labels.float())

    def rerank_pairwise(
        self,
        queries: Tensor,
        documents: Tensor,
        top_k: int = 10,
    ) -> Tuple[Tensor, Tensor]:
        batch_size = queries.size(0)
        num_docs = documents.size(1)

        all_scores = torch.zeros(batch_size, num_docs, device=queries.device)

        for i in range(batch_size):
            query = queries[i]
            for j in range(num_docs):
                doc = documents[:, j, :][:, : documents.size(2)]
                scores = []
                for k in range(num_docs):
                    if k != j:
                        doc_a = documents[:, j, :][:, : documents.size(2)]
                        doc_b = documents[:, k, :][:, : documents.size(2)]
                        score = self.forward(query.unsqueeze(0), doc_a, doc_b)
                        scores.append(score)

                if scores:
                    all_scores[i, j] = torch.stack(scores).mean()

        top_scores, top_indices = torch.topk(all_scores, k=min(top_k, num_docs), dim=1)
        return top_indices, top_scores


class DistillReranker(nn.Module):
    def __init__(
        self,
        vocab_size: int = 30522,
        hidden_dim: int = 768,
        num_layers: int = 6,
        num_heads: int = 8,
        dropout: float = 0.1,
        teacher_dim: Optional[int] = None,
    ):
        super().__init__()

        self.embedding = nn.Embedding(vocab_size, hidden_dim, padding_idx=0)

        self.encoder_layers = nn.ModuleList(
            [
                nn.TransformerEncoderLayer(
                    d_model=hidden_dim,
                    nhead=num_heads,
                    dim_feedforward=hidden_dim * 4,
                    dropout=dropout,
                    activation="gelu",
                    batch_first=True,
                )
                for _ in range(num_layers)
            ]
        )

        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )

        if teacher_dim is not None:
            self.teacher_projection = nn.Linear(teacher_dim, hidden_dim)
        else:
            self.teacher_projection = None

    def forward(self, query_doc_pairs: Tensor) -> Tensor:
        x = self.embedding(query_doc_pairs)

        for layer in self.encoder_layers:
            x = layer(x)

        x = x.mean(dim=1)
        return self.classifier(x)

    def compute_distill_loss(
        self,
        query_doc_pairs: Tensor,
        teacher_scores: Tensor,
        temperature: float = 1.0,
        alpha: float = 0.5,
    ) -> Tensor:
        student_logits = self.forward(query_doc_pairs).squeeze(-1)
        student_probs = F.log_softmax(student_logits / temperature, dim=-1)
        teacher_probs = F.softmax(teacher_scores / temperature, dim=-1)

        kd_loss = F.kl_div(student_probs, teacher_probs, reduction="batchmean")
        kd_loss = kd_loss * (temperature**2)

        return kd_loss * alpha

    def compute_ground_truth_loss(
        self,
        query_doc_pairs: Tensor,
        labels: Tensor,
    ) -> Tensor:
        logits = self.forward(query_doc_pairs).squeeze(-1)
        return F.binary_cross_entropy_with_logits(logits, labels.float())

    def compute_combined_loss(
        self,
        query_doc_pairs: Tensor,
        teacher_scores: Tensor,
        labels: Tensor,
        temperature: float = 1.0,
        alpha: float = 0.5,
    ) -> Tensor:
        distill_loss = self.compute_distill_loss(
            query_doc_pairs, teacher_scores, temperature, alpha
        )
        gt_loss = self.compute_ground_truth_loss(query_doc_pairs, labels)

        return alpha * distill_loss + (1 - alpha) * gt_loss


class ListwiseReranker(nn.Module):
    def __init__(
        self,
        vocab_size: int = 30522,
        hidden_dim: int = 768,
        num_layers: int = 6,
        num_heads: int = 8,
        dropout: float = 0.1,
    ):
        super().__init__()

        self.embedding = nn.Embedding(vocab_size, hidden_dim, padding_idx=0)

        self.encoder_layers = nn.ModuleList(
            [
                nn.TransformerEncoderLayer(
                    d_model=hidden_dim,
                    nhead=num_heads,
                    dim_feedforward=hidden_dim * 4,
                    dropout=dropout,
                    activation="gelu",
                    batch_first=True,
                )
                for _ in range(num_layers)
            ]
        )

        self.scorer = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, query: Tensor, documents: Tensor) -> Tensor:
        batch_size = query.size(0)
        num_docs = documents.size(1)
        seq_len = documents.size(2)

        query_expanded = query.unsqueeze(1).expand(-1, num_docs, -1)
        combined = torch.cat([query_expanded, documents], dim=2)

        x = self.embedding(combined)

        for layer in self.encoder_layers:
            x = layer(x)

        x = x.mean(dim=2)
        scores = self.scorer(x).squeeze(-1)

        return scores

    def rerank(
        self,
        queries: Tensor,
        documents: Tensor,
        top_k: int = 10,
    ) -> Tuple[Tensor, Tensor]:
        scores = self.forward(queries, documents)
        top_scores, top_indices = torch.topk(
            scores, k=min(top_k, scores.size(1)), dim=1
        )
        return top_indices, top_scores
