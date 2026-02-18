"""
Cross-Modal Retrieval for fishstick

This module provides cross-modal retrieval models:
- Dual encoder retrieval
- Cross-modal hashing
- Semantic retrieval
"""

from typing import Optional, Tuple, List
import torch
from torch import Tensor, nn
import torch.nn.functional as F


class DualEncoderRetriever(nn.Module):
    """Dual encoder for cross-modal retrieval."""

    def __init__(
        self,
        source_dim: int,
        target_dim: int,
        embed_dim: int = 256,
        dropout: float = 0.2,
    ):
        super().__init__()
        self.source_projection = nn.Sequential(
            nn.Linear(source_dim, embed_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim, embed_dim),
        )

        self.target_projection = nn.Sequential(
            nn.Linear(target_dim, embed_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim, embed_dim),
        )

    def encode_source(self, source: Tensor) -> Tensor:
        source_emb = self.source_projection(source)
        return F.normalize(source_emb, dim=-1)

    def encode_target(self, target: Tensor) -> Tensor:
        target_emb = self.target_projection(target)
        return F.normalize(target_emb, dim=-1)

    def forward(
        self,
        source: Tensor,
        target: Tensor,
    ) -> Tuple[Tensor, Tensor]:
        source_emb = self.encode_source(source)
        target_emb = self.encode_target(target)
        return source_emb, target_emb


class CrossModalHasher(nn.Module):
    """Cross-modal hashing for efficient retrieval."""

    def __init__(
        self,
        source_dim: int,
        target_dim: int,
        hash_dim: int = 64,
        dropout: float = 0.2,
    ):
        super().__init__()
        self.hash_dim = hash_dim

        self.source_hasher = nn.Sequential(
            nn.Linear(source_dim, hash_dim * 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hash_dim * 2, hash_dim),
        )

        self.target_hasher = nn.Sequential(
            nn.Linear(target_dim, hash_dim * 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hash_dim * 2, hash_dim),
        )

    def hash_source(self, source: Tensor) -> Tensor:
        h = self.source_hasher(source)
        return torch.tanh(h)

    def hash_target(self, target: Tensor) -> Tensor:
        h = self.target_hasher(target)
        return torch.tanh(h)

    def forward(
        self,
        source: Tensor,
        target: Tensor,
    ) -> Tuple[Tensor, Tensor]:
        source_hash = self.hash_source(source)
        target_hash = self.hash_target(target)
        return source_hash, target_hash


class RetrievalRanker(nn.Module):
    """Ranking module for cross-modal retrieval."""

    def __init__(
        self,
        embed_dim: int,
        hidden_dim: int = 256,
        num_layers: int = 2,
        dropout: float = 0.2,
    ):
        super().__init__()
        self.similarity_network = nn.Sequential(
            nn.Linear(embed_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            *[
                nn.Sequential(
                    nn.Linear(hidden_dim, hidden_dim),
                    nn.ReLU(),
                    nn.Dropout(dropout),
                )
                for _ in range(num_layers - 1)
            ],
            nn.Linear(hidden_dim, 1),
        )

    def forward(
        self,
        source_emb: Tensor,
        target_emb: Tensor,
    ) -> Tensor:
        combined = torch.cat([source_emb, target_emb], dim=-1)
        scores = self.similarity_network(combined)
        return scores.squeeze(-1)


class SemanticRetriever(nn.Module):
    """Semantic-aware cross-modal retriever."""

    def __init__(
        self,
        source_dim: int,
        target_dim: int,
        embed_dim: int = 256,
        num_semantic_classes: int = 20,
        dropout: float = 0.2,
    ):
        super().__init__()
        self.retriever = DualEncoderRetriever(
            source_dim, target_dim, embed_dim, dropout
        )

        self.semantic_classifier = nn.Sequential(
            nn.Linear(embed_dim, num_semantic_classes),
        )

    def forward(
        self,
        source: Tensor,
        target: Tensor,
    ) -> Tuple[Tensor, Tensor, Tensor]:
        source_emb, target_emb = self.retriever(source, target)

        source_logits = self.semantic_classifier(source_emb)
        target_logits = self.semantic_classifier(target_emb)

        return source_emb, target_emb, source_logits


class CrossModalAttentionRetrieval(nn.Module):
    """Cross-modal attention for retrieval."""

    def __init__(
        self,
        source_dim: int,
        target_dim: int,
        embed_dim: int = 256,
        num_heads: int = 8,
        num_layers: int = 2,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.source_projection = nn.Linear(source_dim, embed_dim)
        self.target_projection = nn.Linear(target_dim, embed_dim)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=embed_dim * 4,
            dropout=dropout,
            batch_first=True,
        )
        self.cross_attention = nn.TransformerEncoder(encoder_layer, num_layers)

    def forward(
        self,
        source: Tensor,
        target: Tensor,
    ) -> Tuple[Tensor, Tensor]:
        source_emb = self.source_projection(source).unsqueeze(1)
        target_emb = self.target_projection(target).unsqueeze(1)

        source_attended = self.cross_attention(source_emb)
        target_attended = self.cross_attention(target_emb)

        return source_attended.squeeze(1), target_attended.squeeze(1)


class RetrievalLoss(nn.Module):
    """Loss functions for retrieval."""

    def __init__(
        self,
        margin: float = 0.2,
        loss_type: str = "triplet",
    ):
        super().__init__()
        self.margin = margin
        self.loss_type = loss_type

    def forward(
        self,
        query: Tensor,
        positive: Tensor,
        negatives: Tensor,
    ) -> Tensor:
        if self.loss_type == "triplet":
            pos_dist = F.pairwise_distance(query, positive)
            neg_dists = F.pairwise_distance(
                query.unsqueeze(1).expand_as(negatives), negatives
            )
            loss = F.relu(pos_dist.unsqueeze(1) - neg_dists + self.margin)
            return loss.mean()

        elif self.loss_type == "contrastive":
            pos_sim = (query * positive).sum(dim=-1)
            neg_sim = (query.unsqueeze(1) * negatives).sum(dim=-1)
            loss = F.relu(neg_sim - pos_sim.unsqueeze(1) + self.margin)
            return loss.mean()

        elif self.loss_type == "info_nce":
            all_targets = torch.cat([positive, negatives], dim=0)
            all_embeddings = torch.cat([query, query], dim=0)

            similarity = all_embeddings @ all_targets.t() / 0.1

            labels = torch.arange(len(query), device=query.device)
            loss = F.cross_entropy(similarity, labels)
            return loss

        else:
            raise ValueError(f"Unknown loss type: {self.loss_type}")


class CMRLoss(nn.Module):
    """Cross-Modal Ranking loss."""

    def __init__(self, margin: float = 0.1):
        super().__init__()
        self.margin = margin

    def forward(
        self,
        source_emb: Tensor,
        target_emb: Tensor,
        labels: Tensor,
    ) -> Tensor:
        similarity = source_emb @ target_emb.t()

        pos_sim = torch.gather(similarity, 1, labels.unsqueeze(1))
        max_neg_sim = similarity.masked_fill(
            torch.eye(similarity.size(0), device=similarity.device).bool(),
            float("-inf"),
        ).max(dim=1)[0]

        loss = F.relu(max_neg_sim - pos_sim.squeeze() + self.margin)
        return loss.mean()


class HardNegativeMining:
    """Hard negative mining for retrieval."""

    def __init__(
        self,
        strategy: str = "semihard",
        num_negatives: int = 5,
    ):
        self.strategy = strategy
        self.num_negatives = num_negatives

    def mine(
        self,
        query: Tensor,
        positive: Tensor,
        negatives: Tensor,
    ) -> Tuple[Tensor, Tensor, Tensor]:
        pos_sim = (query * positive).sum(dim=-1, keepdim=True)
        neg_sim = (query.unsqueeze(1) * negatives).sum(dim=-1)

        if self.strategy == "hard":
            _, indices = neg_sim.topk(self.num_negatives, dim=1)
            hard_negatives = negatives.gather(1, indices)
            return positive, hard_negatives

        elif self.strategy == "semihard":
            diff = neg_sim - pos_sim
            mask = (diff > -self.margin) & (diff < self.margin)
            hard_negatives = negatives[mask]
            return positive, hard_negatives

        else:
            return positive, negatives


def create_retriever(
    retriever_type: str = "dual",
    **kwargs,
) -> nn.Module:
    """Factory function to create retrievers."""
    if retriever_type == "dual":
        return DualEncoderRetriever(**kwargs)
    elif retriever_type == "hashing":
        return CrossModalHasher(**kwargs)
    elif retriever_type == "semantic":
        return SemanticRetriever(**kwargs)
    elif retriever_type == "attention":
        return CrossModalAttentionRetrieval(**kwargs)
    else:
        raise ValueError(f"Unknown retriever type: {retriever_type}")
