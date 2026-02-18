"""
Image-Text Matching for fishstick

This module provides image-text matching and retrieval models:
- Dual encoder matching
- Cross-attention matching
- Ranking and similarity computation
"""

from typing import Optional, Tuple
import torch
from torch import Tensor, nn
import torch.nn.functional as F


class DualEncoderMatcher(nn.Module):
    """Dual encoder for image-text matching."""

    def __init__(
        self,
        image_dim: int = 512,
        text_dim: int = 512,
        embed_dim: int = 256,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.image_projection = nn.Sequential(
            nn.Linear(image_dim, embed_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim, embed_dim),
        )
        self.text_projection = nn.Sequential(
            nn.Linear(text_dim, embed_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim, embed_dim),
        )

    def forward(
        self,
        image_features: Tensor,
        text_features: Tensor,
    ) -> Tuple[Tensor, Tensor]:
        image_emb = self.image_projection(image_features)
        text_emb = self.text_projection(text_features)

        image_emb = F.normalize(image_emb, dim=-1)
        text_emb = F.normalize(text_emb, dim=-1)

        return image_emb, text_emb


class CrossAttentionMatcher(nn.Module):
    """Cross-attention based image-text matching."""

    def __init__(
        self,
        image_dim: int = 512,
        text_dim: int = 512,
        embed_dim: int = 256,
        num_heads: int = 8,
        num_layers: int = 2,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.image_projection = nn.Linear(image_dim, embed_dim)
        self.text_projection = nn.Linear(text_dim, embed_dim)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=embed_dim * 4,
            dropout=dropout,
            batch_first=True,
        )
        self.cross_attention = nn.TransformerEncoder(encoder_layer, num_layers)

        self.classifier = nn.Sequential(
            nn.Linear(embed_dim * 2, embed_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim, 1),
        )

    def forward(
        self,
        image_features: Tensor,
        text_features: Tensor,
        image_mask: Optional[Tensor] = None,
        text_mask: Optional[Tensor] = None,
    ) -> Tensor:
        image_emb = self.image_projection(image_features)
        text_emb = self.text_projection(text_features)

        if image_emb.dim() == 2:
            image_emb = image_emb.unsqueeze(1)
        if text_emb.dim() == 2:
            text_emb = text_emb.unsqueeze(1)

        combined = torch.cat([image_emb, text_emb], dim=1)
        features = self.cross_attention(combined)

        image_out = features[:, : image_emb.size(1), :].mean(dim=1)
        text_out = features[:, image_emb.size(1) :, :].mean(dim=1)

        joint = torch.cat([image_out, text_out], dim=-1)
        score = self.classifier(joint)

        return score.squeeze(-1)


class SimilarityMatrix(nn.Module):
    """Compute similarity matrix between image and text embeddings."""

    def __init__(
        self,
        temperature: float = 0.1,
        learnable_temperature: bool = False,
    ):
        super().__init__()
        if learnable_temperature:
            self.logit_scale = nn.Parameter(
                torch.ones([]) * torch.log(torch.tensor(1.0 / temperature))
            )
        else:
            self.register_buffer("logit_scale", torch.tensor(1.0 / temperature))

    def forward(self, image_emb: Tensor, text_emb: Tensor) -> Tensor:
        return self.logit_scale * image_emb @ text_emb.t()


class RankingLoss(nn.Module):
    """Ranking loss for image-text retrieval."""

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
        anchors: Tensor,
        positives: Tensor,
        negatives: Tensor,
    ) -> Tensor:
        if self.loss_type == "triplet":
            pos_dist = F.pairwise_distance(anchors, positives)
            neg_dist = F.pairwise_distance(anchors, negatives)
            loss = F.relu(pos_dist - neg_dist + self.margin)
            return loss.mean()
        elif self.loss_type == "contrastive":
            pos_sim = (anchors * positives).sum(dim=-1)
            neg_sim = (anchors * negatives).sum(dim=-1)
            loss = F.relu(neg_sim - pos_sim + self.margin)
            return loss.mean()
        else:
            raise ValueError(f"Unknown loss type: {self.loss_type}")


class ImageTextRetrieval(nn.Module):
    """Complete image-text retrieval system."""

    def __init__(
        self,
        image_encoder: nn.Module,
        text_encoder: nn.Module,
        embed_dim: int = 256,
        temperature: float = 0.07,
    ):
        super().__init__()
        self.image_encoder = image_encoder
        self.text_encoder = text_encoder

        self.matcher = DualEncoderMatcher(
            image_dim=embed_dim,
            text_dim=embed_dim,
            embed_dim=embed_dim,
        )
        self.similarity = SimilarityMatrix(
            temperature=temperature, learnable_temperature=True
        )

    def encode_image(self, image: Tensor) -> Tensor:
        image_features = self.image_encoder(image)
        image_emb, _ = self.matcher(image_features, torch.zeros_like(image_features))
        return image_emb

    def encode_text(self, text: Tensor) -> Tensor:
        text_features = self.text_encoder(text)
        _, text_emb = self.matcher(torch.zeros_like(text_features), text_features)
        return text_emb

    def forward(
        self,
        image: Tensor,
        text: Tensor,
    ) -> Tuple[Tensor, Tensor]:
        image_features = self.image_encoder(image)
        text_features = self.text_encoder(text)

        image_emb, text_emb = self.matcher(image_features, text_features)

        logits = self.similarity(image_emb, text_emb)

        return logits, image_emb


class HardNegativeMiner(nn.Module):
    """Hard negative mining for retrieval."""

    def __init__(
        self,
        mining_strategy: str = "semihard",
        margin: float = 0.2,
    ):
        super().__init__()
        self.mining_strategy = mining_strategy
        self.margin = margin

    def forward(
        self,
        anchors: Tensor,
        positives: Tensor,
        negatives: Tensor,
    ) -> Tuple[Tensor, Tensor]:
        pos_sim = (anchors * positives).sum(dim=-1)
        neg_sim = (anchors * negatives).sum(dim=-1)

        if self.mining_strategy == "semihard":
            mask = (neg_sim > pos_sim - self.margin) & (neg_sim < pos_sim + self.margin)
        elif self.mining_strategy == "hard":
            mask = neg_sim > pos_sim
        else:
            mask = torch.ones_like(neg_sim, dtype=torch.bool)

        hard_negatives = negatives[mask]
        return positives[mask], hard_negatives


def create_matcher(
    matcher_type: str = "dual",
    **kwargs,
) -> nn.Module:
    """Factory function to create a matcher."""
    if matcher_type == "dual":
        return DualEncoderMatcher(**kwargs)
    elif matcher_type == "cross_attention":
        return CrossAttentionMatcher(**kwargs)
    else:
        raise ValueError(f"Unknown matcher type: {matcher_type}")
