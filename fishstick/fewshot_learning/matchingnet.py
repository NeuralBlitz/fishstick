"""
Matching Networks for few-shot learning.

Matching Networks use an attention mechanism to compare query examples
to support set examples for classification.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from typing import Optional, Dict, List, Tuple

from .config import MatchingNetworkConfig
from .types import FewShotTask


class MatchingNetwork(nn.Module):
    """Matching Networks for few-shot learning.

    Uses attention over the support set to classify query examples.

    Args:
        encoder: Feature encoder network
        config: Matching network configuration

    Example:
        >>> encoder = nn.Sequential(nn.Conv2d(3, 64, 3), nn.ReLU(), nn.AdaptiveAvgPool2d(1))
        >>> mnet = MatchingNetwork(encoder, MatchingNetworkConfig(attention='cosine'))
        >>> task = FewShotTask(support_x, support_y, query_x, query_y, 5, 5, 15)
        >>> logits = mnet(task)

    References:
        Vinyals et al. "Matching Networks for One Shot Learning" (NeurIPS 2016)
    """

    def __init__(
        self,
        encoder: nn.Module,
        config: Optional[MatchingNetworkConfig] = None,
    ):
        super().__init__()
        self.encoder = encoder
        self.config = config or MatchingNetworkConfig()

        self._embedding_dim = self._get_embedding_dim()

        self._build_attention()

    def _get_embedding_dim(self) -> int:
        self.encoder.eval()
        with torch.no_grad():
            dummy = torch.zeros(1, 3, 84, 84)
            if torch.cuda.is_available():
                dummy = dummy.cuda()
            out = self.encoder(dummy)
        self.encoder.train()
        return out.view(out.size(0), -1).size(1)

    def _build_attention(self) -> None:
        """Build attention module based on config."""
        if self.config.attention == "cosine":
            self.scale = nn.Parameter(torch.ones(1) * 10.0)
        elif self.config.attention == "embedded":
            self.g = nn.Sequential(
                nn.Linear(self._embedding_dim, self._embedding_dim),
                nn.ReLU(),
                nn.Linear(self._embedding_dim, self._embedding_dim),
            )
            self.f = nn.Sequential(
                nn.Linear(self._embedding_dim, self._embedding_dim),
                nn.ReLU(),
                nn.Linear(self._embedding_dim, self._embedding_dim),
            )

        if self.config.full_context and self.config.lstm_layers > 0:
            self.lstm = nn.LSTM(
                input_size=self._embedding_dim,
                hidden_size=self._embedding_dim,
                num_layers=self.config.lstm_layers,
                batch_first=True,
                dropout=self.config.lstm_dropout if self.config.lstm_layers > 1 else 0,
            )

    def _encode(self, x: Tensor) -> Tensor:
        """Encode inputs to feature embeddings."""
        x = x.view(-1, *x.shape[2:])
        emb = self.encoder(x)
        return emb.view(emb.size(0), -1)

    def _compute_attention(
        self,
        query_emb: Tensor,
        support_emb: Tensor,
    ) -> Tensor:
        """Compute attention weights between queries and support set.

        Args:
            query_emb: Query embeddings [n_query, embedding_dim]
            support_emb: Support embeddings [n_support, embedding_dim]

        Returns:
            Attention weights [n_query, n_support]
        """
        if self.config.attention == "cosine":
            query_norm = F.normalize(query_emb, p=2, dim=1)
            support_norm = F.normalize(support_emb, p=2, dim=1)

            similarity = torch.mm(query_norm, support_norm.t())
            attention = F.softmax(self.scale * similarity, dim=1)

        elif self.config.attention == "dot":
            similarity = torch.mm(query_emb, support_emb.t())
            attention = F.softmax(similarity, dim=1)

        elif self.config.attention == "embedded":
            g_emb = self.g(support_emb)
            f_emb = self.f(query_emb)

            similarity = torch.mm(f_emb, g_emb.t())
            attention = F.softmax(similarity, dim=1)

        else:
            raise ValueError(f"Unknown attention: {self.config.attention}")

        return attention

    def forward(
        self,
        support_x: Tensor,
        support_y: Tensor,
        query_x: Tensor,
        n_way: int,
        n_shot: int,
    ) -> Tensor:
        """Compute query predictions using attention over support set.

        Args:
            support_x: Support set inputs
            support_y: Support set labels
            query_x: Query set inputs
            n_way: Number of classes
            n_shot: Number of support examples per class

        Returns:
            Logits [n_query, n_way]
        """
        support_emb = self._encode(support_x)
        query_emb = self._encode(query_x)

        if self.config.full_context and hasattr(self, "lstm"):
            support_emb, _ = self.lstm(support_emb.unsqueeze(0))
            support_emb = support_emb.squeeze(0)

        attention = self._compute_attention(query_emb, support_emb)

        n_query = query_emb.size(0)

        label_map = torch.zeros(
            n_query, n_way, device=support_x.device, dtype=torch.float32
        )

        for c in range(n_way):
            class_mask = support_y == c
            label_map[:, c] = attention[:, class_mask].sum(1)

        return label_map

    def __call__(self, task: FewShotTask) -> Tensor:
        """Forward pass using FewShotTask."""
        return self.forward(
            task.support_x,
            task.support_y,
            task.query_x,
            task.n_way,
            task.n_shot,
        )


class FullContextMatchingNetwork(MatchingNetwork):
    """Matching Networks with full context embedding (fce)."""

    def __init__(
        self,
        encoder: nn.Module,
        config: Optional[MatchingNetworkConfig] = None,
    ):
        if config is None:
            config = MatchingNetworkConfig(full_context=True, lstm_layers=2)
        else:
            config.full_context = True
            config.lstm_layers = max(config.lstm_layers, 1)

        super().__init__(encoder, config)


class ConvolutionalMatchingNetwork(MatchingNetwork):
    """Matching Networks with convolutional context encoding."""

    def __init__(
        self,
        encoder: nn.Module,
        config: Optional[MatchingNetworkConfig] = None,
    ):
        super().__init__(encoder, config)

        self.context_conv = nn.Sequential(
            nn.Conv1d(
                self._embedding_dim, self._embedding_dim, kernel_size=3, padding=1
            ),
            nn.ReLU(),
            nn.Conv1d(
                self._embedding_dim, self._embedding_dim, kernel_size=3, padding=1
            ),
        )

    def forward(
        self,
        support_x: Tensor,
        support_y: Tensor,
        query_x: Tensor,
        n_way: int,
        n_shot: int,
    ) -> Tensor:
        """Compute with convolutional context encoding."""
        support_emb = self._encode(support_x)
        query_emb = self._encode(query_x)

        n_support = support_emb.size(0)
        support_reshaped = support_emb.T.unsqueeze(0)

        context = self.context_conv(support_reshaped)
        context = context.squeeze(0).T

        attention = self._compute_attention(query_emb, context)

        n_query = query_emb.size(0)

        label_map = torch.zeros(
            n_query, n_way, device=support_x.device, dtype=torch.float32
        )

        for c in range(n_way):
            class_mask = support_y == c
            label_map[:, c] = attention[:, class_mask].sum(1)

        return label_map


class ImprintedWeights(nn.Module):
    """Imprinting weights for few-shot classification.

    Imprints class weights directly from support set embeddings.

    Args:
        encoder: Feature encoder
        num_classes: Number of classes to imprint
    """

    def __init__(
        self,
        encoder: nn.Module,
        num_classes: int,
    ):
        super().__init__()
        self.encoder = encoder
        self.num_classes = num_classes
        self._embedding_dim = self._get_embedding_dim()

        self.weight = nn.Parameter(torch.randn(num_classes, self._embedding_dim))

    def _get_embedding_dim(self) -> int:
        self.encoder.eval()
        with torch.no_grad():
            dummy = torch.zeros(1, 3, 84, 84)
            if torch.cuda.is_available():
                dummy = dummy.cuda()
            out = self.encoder(dummy)
        self.encoder.train()
        return out.view(out.size(0), -1).size(1)

    def imprint(
        self,
        support_x: Tensor,
        support_y: Tensor,
    ) -> None:
        """Imprint class weights from support set.

        Args:
            support_x: Support set inputs
            support_y: Support set labels
        """
        self.encoder.eval()
        with torch.no_grad():
            support_emb = self.encoder(support_x)
            support_emb = support_emb.view(support_emb.size(0), -1)
            support_emb = F.normalize(support_emb, p=2, dim=1)

        unique_classes = torch.unique(support_y)

        for c in unique_classes:
            class_mask = support_y == c
            class_emb = support_emb[class_mask]

            self.weight.data[c] = class_emb.mean(0)

    def forward(self, x: Tensor) -> Tensor:
        """Compute logits using imprinted weights."""
        x = x.view(-1, *x.shape[2:])
        emb = self.encoder(x)
        emb = emb.view(emb.size(0), -1)
        emb = F.normalize(emb, p=2, dim=1)

        weight = F.normalize(self.weight, p=2, dim=1)

        logits = torch.mm(emb, weight.t())

        return logits
