"""
Relation Networks for few-shot learning.

Relation Networks learn a non-linear distance function between query
embeddings and class prototypes using a relation module.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from typing import Optional, Dict, List, Tuple

from .config import RelationNetworkConfig
from .types import FewShotTask


class RelationNetwork(nn.Module):
    """Relation Network for few-shot learning.

    Learns to compare query examples to class prototypes using a
    learnable relation module.

    Args:
        encoder: Feature encoder network
        config: Relation network configuration

    Example:
        >>> encoder = nn.Sequential(nn.Conv2d(3, 64, 3), nn.ReLU(), nn.AdaptiveAvgPool2d(1))
        >>> relnet = RelationNetwork(encoder, RelationNetworkConfig(relation_dim=8, num_relation_layers=3))
        >>> task = FewShotTask(support_x, support_y, query_x, query_y, 5, 5, 15)
        >>> logits = relnet(task)

    References:
        Sung et al. "Learning to Compare: Relation Network for Few-Shot Learning" (CVPR 2018)
    """

    def __init__(
        self,
        encoder: nn.Module,
        config: Optional[RelationNetworkConfig] = None,
    ):
        super().__init__()
        self.encoder = encoder
        self.config = config or RelationNetworkConfig()

        self._embedding_dim = self._get_embedding_dim()

        self.relation_module = self._build_relation_module()

    def _get_embedding_dim(self) -> int:
        self.encoder.eval()
        with torch.no_grad():
            dummy = torch.zeros(1, 3, 84, 84)
            if torch.cuda.is_available():
                dummy = dummy.cuda()
            out = self.encoder(dummy)
        self.encoder.train()
        return out.view(out.size(0), -1).size(1)

    def _build_relation_module(self) -> nn.Module:
        """Build the relation module for computing similarity."""
        input_dim = self._embedding_dim * 2

        layers = []
        for i in range(self.config.num_relation_layers):
            layers.append(nn.Linear(input_dim, self.config.relation_dim))
            if self.config.activation == "relu":
                layers.append(nn.ReLU())
            elif self.config.activation == "leaky_relu":
                layers.append(nn.LeakyReLU())
            elif self.config.activation == "gelu":
                layers.append(nn.GELU())

            if self.config.dropout > 0:
                layers.append(nn.Dropout(self.config.dropout))

            input_dim = self.config.relation_dim

        layers.append(nn.Linear(self.config.relation_dim, 1))
        layers.append(nn.Sigmoid())

        return nn.Sequential(*layers)

    def _encode(self, x: Tensor) -> Tensor:
        """Encode inputs to feature embeddings."""
        x = x.view(-1, *x.shape[2:])
        emb = self.encoder(x)
        return emb.view(emb.size(0), -1)

    def _compute_prototypes(
        self,
        support_emb: Tensor,
        support_y: Tensor,
        n_way: int,
        n_shot: int,
    ) -> Tensor:
        """Compute class prototypes as mean of support embeddings."""
        prototypes = torch.zeros(n_way, support_emb.size(1), device=support_emb.device)

        for c in range(n_way):
            class_mask = support_y == c
            prototypes[c] = support_emb[class_mask].mean(0)

        return prototypes

    def forward(
        self,
        support_x: Tensor,
        support_y: Tensor,
        query_x: Tensor,
        n_way: int,
        n_shot: int,
    ) -> Tensor:
        """Compute relation scores between queries and class prototypes.

        Args:
            support_x: Support set inputs
            support_y: Support set labels
            query_x: Query set inputs
            n_way: Number of classes
            n_shot: Number of support examples per class

        Returns:
            Relation scores [n_query, n_way]
        """
        support_emb = self._encode(support_x)
        query_emb = self._encode(query_x)

        prototypes = self._compute_prototypes(support_emb, support_y, n_way, n_shot)

        n_query = query_emb.size(0)

        query_expanded = query_emb.unsqueeze(1).expand(n_query, n_way, -1)
        prototype_expanded = prototypes.unsqueeze(0).expand(n_query, -1, -1)

        relation_pairs = torch.cat([query_expanded, prototype_expanded], dim=2)

        relations = self.relation_module(relation_pairs).squeeze(2)

        return relations

    def __call__(self, task: FewShotTask) -> Tensor:
        """Forward pass using FewShotTask."""
        return self.forward(
            task.support_x,
            task.support_y,
            task.query_x,
            task.n_way,
            task.n_shot,
        )


class DeepRelationNetwork(RelationNetwork):
    """Deep Relation Network with deeper feature encoder."""

    def __init__(
        self,
        encoder: nn.Module,
        config: Optional[RelationNetworkConfig] = None,
    ):
        super().__init__(encoder, config)

        if config is None:
            self.config.num_relation_layers = 4
            self.config.relation_dim = 128


class MultiScaleRelationNetwork(RelationNetwork):
    """Multi-scale Relation Network with feature pyramid."""

    def __init__(
        self,
        encoder: nn.Module,
        config: Optional[RelationNetworkConfig] = None,
        scales: List[int] = [1, 2, 4],
    ):
        super().__init__(encoder, config)
        self.scales = scales

        self.scale_projections = nn.ModuleList(
            [
                nn.Linear(self._embedding_dim, self._embedding_dim // len(scales))
                for _ in scales
            ]
        )

    def _encode_multiscale(self, x: Tensor) -> List[Tensor]:
        """Encode inputs at multiple scales."""
        base_emb = self._encode(x)

        scale_embs = []
        for i, scale in enumerate(self.scales):
            if scale == 1:
                scale_emb = base_emb
            else:
                b, c, h, w = x.shape
                h_scaled, w_scaled = max(1, h // scale), max(1, w // scale)
                x_scaled = F.adaptive_avg_pool2d(
                    x.view(b, c, h, w), (h_scaled, w_scaled)
                )
                scale_emb = self.encoder(x_scaled.view(-1, c, h_scaled, w_scaled))
                scale_emb = scale_emb.view(scale_emb.size(0), -1)

            scale_embs.append(self.scale_projections[i](scale_emb))

        return scale_embs

    def forward(
        self,
        support_x: Tensor,
        support_y: Tensor,
        query_x: Tensor,
        n_way: int,
        n_shot: int,
    ) -> Tensor:
        """Compute multi-scale relation scores."""
        support_embs = self._encode_multiscale(support_x)
        query_embs = self._encode_multiscale(query_x)

        support_combined = torch.cat(support_embs, dim=1)
        query_combined = torch.cat(query_embs, dim=1)

        return super().forward(
            support_combined,
            support_y,
            query_combined,
            n_way,
            n_shot,
        )


class AttentionRelationNetwork(RelationNetwork):
    """Relation Network with attention-based comparison."""

    def __init__(
        self,
        encoder: nn.Module,
        config: Optional[RelationNetworkConfig] = None,
    ):
        super().__init__(encoder, config)

        self.attention = nn.MultiheadAttention(
            embed_dim=self._embedding_dim,
            num_heads=4,
            batch_first=True,
        )

    def forward(
        self,
        support_x: Tensor,
        support_y: Tensor,
        query_x: Tensor,
        n_way: int,
        n_shot: int,
    ) -> Tensor:
        """Compute attention-weighted relations."""
        support_emb = self._encode(support_x)
        query_emb = self._encode(query_x)

        prototypes = self._compute_prototypes(support_emb, support_y, n_way, n_shot)

        query_prototype_pairs = torch.stack(
            [query_emb, prototypes.expand(query_emb.size(0), -1, -1)], dim=1
        )

        attended, _ = self.attention(
            query_prototype_pairs, query_prototype_pairs, query_prototype_pairs
        )

        relation_input = attended.mean(1)

        n_query = query_emb.size(0)
        relation_input = relation_input.view(n_query, -1)

        return self.relation_module(relation_input).squeeze(1)
