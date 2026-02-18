"""
Prototypical Networks implementation for few-shot learning.

Prototypical Networks compute a prototype (mean) for each class in the
support set and classify query points based on distance to prototypes.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from typing import Optional, Dict, List, Tuple
import math

from .config import PrototypicalConfig
from .types import FewShotTask


class PrototypicalNetworks(nn.Module):
    """Prototypical Networks for few-shot learning.

    Computes class prototypes as the mean of support set embeddings,
    then classifies query points based on distance to prototypes.

    Args:
        encoder: Feature encoder network
        config: Prototypical configuration

    Example:
        >>> encoder = nn.Sequential(nn.Conv2d(3, 64, 3), nn.ReLU(), nn.AdaptiveAvgPool2d(1))
        >>> protonet = PrototypicalNetworks(encoder, PrototypicalConfig(distance='euclidean'))
        >>> task = FewShotTask(support_x, support_y, query_x, query_y, 5, 5, 15)
        >>> logits = protonet(task)

    References:
        Snell et al. "Prototypical Networks for Few-shot Learning" (NeurIPS 2017)
    """

    def __init__(
        self,
        encoder: nn.Module,
        config: Optional[PrototypicalConfig] = None,
    ):
        super().__init__()
        self.encoder = encoder
        self.config = config or PrototypicalConfig()

        self._embedding_dim: Optional[int] = None

    @property
    def embedding_dim(self) -> int:
        if self._embedding_dim is None:
            self._embedding_dim = self._get_embedding_dim()
        return self._embedding_dim

    def _get_embedding_dim(self) -> int:
        """Get embedding dimension by forward pass."""
        self.encoder.eval()
        with torch.no_grad():
            dummy = torch.zeros(1, 3, 84, 84)
            if torch.cuda.is_available():
                dummy = dummy.cuda()
            out = self.encoder(dummy)
        self.encoder.train()
        return out.view(out.size(0), -1).size(1)

    def forward(
        self,
        support_x: Tensor,
        support_y: Tensor,
        query_x: Tensor,
        n_way: int,
        n_shot: int,
    ) -> Tuple[Tensor, Tensor]:
        """Compute query predictions using prototypical networks.

        Args:
            support_x: Support set inputs [n_way * n_shot, ...]
            support_y: Support set labels [n_way * n_shot]
            query_x: Query set inputs [n_way * n_query, ...]
            n_way: Number of classes
            n_shot: Number of support examples per class

        Returns:
            Tuple of (query_logits, prototypes)
            - query_logits: [n_query, n_way]
            - prototypes: [n_way, embedding_dim]
        """
        support_emb = self._encode(support_x)
        query_emb = self._encode(query_x)

        prototypes = self._compute_prototypes(support_emb, support_y, n_way)

        logits = self._compute_logits(query_emb, prototypes)

        return logits, prototypes

    def _encode(self, x: Tensor) -> Tensor:
        """Encode inputs to feature embeddings."""
        x = x.view(-1, *x.shape[2:])
        emb = self.encoder(x)
        emb = emb.view(emb.size(0), -1)

        if self.config.normalized:
            emb = F.normalize(emb, p=2, dim=1)

        return emb

    def _compute_prototypes(
        self,
        support_emb: Tensor,
        support_y: Tensor,
        n_way: int,
    ) -> Tensor:
        """Compute class prototypes as mean of support embeddings."""
        prototypes = torch.zeros(n_way, support_emb.size(1), device=support_emb.device)

        for c in range(n_way):
            class_mask = support_y == c
            if class_mask.sum() > 0:
                prototypes[c] = support_emb[class_mask].mean(0)

        return prototypes

    def _compute_logits(self, query_emb: Tensor, prototypes: Tensor) -> Tensor:
        """Compute query logits based on distance to prototypes."""
        if self.config.distance == "euclidean":
            if self.config.squared:
                logits = -(torch.cdist(query_emb, prototypes, p=2) ** 2)
            else:
                logits = -torch.cdist(query_emb, prototypes, p=2)
        elif self.config.distance == "cosine":
            logits = torch.mm(query_emb, prototypes.t())
        elif self.config.distance == "manhattan":
            logits = -torch.cdist(query_emb, prototypes, p=1)
        else:
            raise ValueError(f"Unknown distance: {self.config.distance}")

        if self.config.temperature != 1.0:
            logits = logits / self.config.temperature

        return logits

    def __call__(
        self,
        task: FewShotTask,
    ) -> Tuple[Tensor, Tensor]:
        """Forward pass using FewShotTask."""
        return self.forward(
            task.support_x,
            task.support_y,
            task.query_x,
            task.n_way,
            task.n_shot,
        )


class SoftPrototypicalNetworks(PrototypicalNetworks):
    """Soft Prototypical Networks with attention-weighted prototypes.

    Uses attention to weight support examples when computing prototypes.
    """

    def __init__(
        self,
        encoder: nn.Module,
        config: Optional[PrototypicalConfig] = None,
        attention_dim: int = 64,
    ):
        super().__init__(encoder, config)
        self.attention_dim = attention_dim

        self.prototype_attention = nn.Sequential(
            nn.Linear(self.embedding_dim, attention_dim),
            nn.Tanh(),
            nn.Linear(attention_dim, 1),
        )

    def _compute_prototypes(
        self,
        support_emb: Tensor,
        support_y: Tensor,
        n_way: int,
    ) -> Tensor:
        """Compute attention-weighted prototypes."""
        prototypes = torch.zeros(n_way, support_emb.size(1), device=support_emb.device)

        for c in range(n_way):
            class_mask = support_y == c
            class_emb = support_emb[class_mask]

            if class_emb.size(0) > 0:
                attn_weights = self.prototype_attention(class_emb)
                attn_weights = F.softmax(attn_weights, dim=0)

                prototypes[c] = (class_emb * attn_weights).sum(0)

        return prototypes


class VariationalPrototypicalNetworks(PrototypicalNetworks):
    """Variational Prototypical Networks with Bayesian inference.

    Models prototypes as latent variables with Gaussian distributions.
    """

    def __init__(
        self,
        encoder: nn.Module,
        config: Optional[PrototypicalConfig] = None,
        latent_dim: int = 64,
    ):
        super().__init__(encoder, config)
        self.latent_dim = latent_dim

        self.mu_head = nn.Linear(self.embedding_dim, latent_dim)
        self.logvar_head = nn.Linear(self.embedding_dim, latent_dim)

    def _compute_prototypes(
        self,
        support_emb: Tensor,
        support_y: Tensor,
        n_way: int,
    ) -> Tuple[Tensor, Tensor]:
        """Compute Gaussian prototype distributions."""
        mu = torch.zeros(n_way, self.latent_dim, device=support_emb.device)
        logvar = torch.zeros(n_way, self.latent_dim, device=support_emb.device)

        for c in range(n_way):
            class_mask = support_y == c
            class_emb = support_emb[class_mask]

            if class_emb.size(0) > 0:
                mu[c] = class_emb.mean(0)
                logvar[c] = torch.log(class_emb.var(0) + 1e-6)

        return mu, logvar

    def _sample_prototypes(self, mu: Tensor, logvar: Tensor) -> Tensor:
        """Reparameterize and sample from prototype distributions."""
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std


def compute_prototypical_accuracy(
    logits: Tensor,
    targets: Tensor,
) -> float:
    """Compute classification accuracy from logits.

    Args:
        logits: Model predictions [n_query, n_way]
        targets: True labels [n_query]

    Returns:
        Accuracy as float
    """
    preds = torch.argmax(logits, dim=1)
    return (preds == targets).float().mean().item()
