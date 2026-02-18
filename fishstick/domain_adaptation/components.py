"""
Reusable Components for Domain Adaptation Module for Fishstick.

This module provides reusable components for building domain adaptation models
including domain classifiers, feature extractors, encoder-decoders, and attention
mechanisms.

Example:
    >>> from fishstick.domain_adaptation.components import DomainClassifier
    >>> classifier = DomainClassifier(input_dim=256, num_domains=2)
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.nn import Module


class DomainClassifier(Module):
    """Domain classifier for adversarial domain adaptation.

    A simple MLP-based binary domain classifier.

    Args:
        input_dim: Dimension of input features.
        hidden_dims: Hidden layer dimensions.
        dropout: Dropout probability.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dims: Optional[List[int]] = None,
        dropout: float = 0.5,
    ):
        super().__init__()

        if hidden_dims is None:
            hidden_dims = [128, 64]

        layers = []
        prev_dim = input_dim

        for hidden_dim in hidden_dims:
            layers.extend(
                [
                    nn.Linear(prev_dim, hidden_dim),
                    nn.ReLU(),
                    nn.Dropout(dropout),
                ]
            )
            prev_dim = hidden_dim

        layers.append(nn.Linear(prev_dim, 1))

        self.classifier = nn.Sequential(*layers)

    def forward(self, x: Tensor) -> Tensor:
        return torch.sigmoid(self.classifier(x))


class FeatureExtractorNetwork(Module):
    """Feature extraction network for domain adaptation.

    Args:
        input_dim: Dimension of input features.
        output_dim: Dimension of output features.
        hidden_dims: Hidden layer dimensions.
        use_bn: Whether to use batch normalization.
        dropout: Dropout probability.
    """

    def __init__(
        self,
        input_dim: int,
        output_dim: int = 128,
        hidden_dims: Optional[List[int]] = None,
        use_bn: bool = True,
        dropout: float = 0.5,
    ):
        super().__init__()

        if hidden_dims is None:
            hidden_dims = [256, 128]

        layers = []
        prev_dim = input_dim

        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            if use_bn:
                layers.append(nn.BatchNorm1d(hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
            prev_dim = hidden_dim

        layers.append(nn.Linear(prev_dim, output_dim))

        self.extractor = nn.Sequential(*layers)
        self.output_dim = output_dim

    def forward(self, x: Tensor) -> Tensor:
        return self.extractor(x)


class EncoderDecoder(Module):
    """Encoder-Decoder architecture for domain adaptation.

    Used in reconstruction-based domain adaptation methods.

    Args:
        input_dim: Dimension of input features.
        latent_dim: Dimension of latent space.
        hidden_dims: Hidden layer dimensions.
        dropout: Dropout probability.
    """

    def __init__(
        self,
        input_dim: int,
        latent_dim: int = 64,
        hidden_dims: Optional[List[int]] = None,
        dropout: float = 0.5,
    ):
        super().__init__()

        if hidden_dims is None:
            hidden_dims = [256, 128]

        encoder_layers = []
        prev_dim = input_dim

        for hidden_dim in hidden_dims:
            encoder_layers.extend(
                [
                    nn.Linear(prev_dim, hidden_dim),
                    nn.BatchNorm1d(hidden_dim),
                    nn.ReLU(),
                    nn.Dropout(dropout),
                ]
            )
            prev_dim = hidden_dim

        encoder_layers.append(nn.Linear(prev_dim, latent_dim))
        self.encoder = nn.Sequential(*encoder_layers)

        decoder_layers = []
        prev_dim = latent_dim

        for hidden_dim in reversed(hidden_dims):
            decoder_layers.extend(
                [
                    nn.Linear(prev_dim, hidden_dim),
                    nn.BatchNorm1d(hidden_dim),
                    nn.ReLU(),
                    nn.Dropout(dropout),
                ]
            )
            prev_dim = hidden_dim

        decoder_layers.append(nn.Linear(prev_dim, input_dim))
        self.decoder = nn.Sequential(*decoder_layers)

        self.latent_dim = latent_dim

    def encode(self, x: Tensor) -> Tensor:
        return self.encoder(x)

    def decode(self, z: Tensor) -> Tensor:
        return self.decoder(z)

    def forward(self, x: Tensor) -> Tuple[Tensor, Tensor]:
        z = self.encode(x)
        x_recon = self.decode(z)
        return z, x_recon


class AttentionDomainAdaptation(Module):
    """Attention-based domain adaptation module.

    Uses attention mechanism to focus on domain-invariant features.

    Args:
        feature_dim: Dimension of input features.
        attention_dim: Dimension of attention hidden layer.
        num_domains: Number of domains (for domain-specific attention).
    """

    def __init__(
        self,
        feature_dim: int,
        attention_dim: int = 64,
        num_domains: int = 2,
    ):
        super().__init__()
        self.feature_dim = feature_dim
        self.attention_dim = attention_dim
        self.num_domains = num_domains

        self.query = nn.Linear(feature_dim, attention_dim)
        self.key = nn.Linear(feature_dim, attention_dim)
        self.value = nn.Linear(feature_dim, attention_dim)

        self.domain_query = nn.Embedding(num_domains, attention_dim)

        self.out_projection = nn.Linear(attention_dim, feature_dim)

    def forward(
        self,
        features: Tensor,
        domain_id: Optional[int] = None,
    ) -> Tuple[Tensor, Tensor]:
        batch_size = features.size(0)

        Q = self.query(features)
        K = self.key(features)
        V = self.value(features)

        if domain_id is not None:
            domain_vector = self.domain_query(
                torch.tensor(domain_id, device=features.device)
            )
            Q = Q + domain_vector.unsqueeze(0)

        attention_scores = torch.matmul(Q, K.transpose(-2, -1)) / (
            self.attention_dim**0.5
        )
        attention_weights = F.softmax(attention_scores, dim=-1)

        attended = torch.matmul(attention_weights, V)
        output = self.out_projection(attended)

        return output, attention_weights


class ResidualAdapter(Module):
    """Residual adapter for domain adaptation.

    Adds domain-specific residual connections to pre-trained models.

    Args:
        feature_dim: Dimension of input features.
        bottleneck_dim: Dimension of bottleneck layer.
    """

    def __init__(
        self,
        feature_dim: int,
        bottleneck_dim: int = 64,
    ):
        super().__init__()

        self.down_project = nn.Linear(feature_dim, bottleneck_dim)
        self.up_project = nn.Linear(bottleneck_dim, feature_dim)

        self.norm = nn.LayerNorm(feature_dim)

    def forward(self, x: Tensor) -> Tensor:
        residual = x
        bottleneck = F.relu(self.down_project(x))
        adapted = self.up_project(bottleneck)
        return self.norm(residual + adapted)


class DomainSpecificBatchNorm2d(nn.Module):
    """Domain-specific 2D batch normalization.

    Maintains separate BN statistics for each domain.

    Args:
        num_features: Number of features.
        num_domains: Number of domains.
        eps: Epsilon for numerical stability.
    """

    def __init__(
        self,
        num_features: int,
        num_domains: int,
        eps: float = 1e-5,
    ):
        super().__init__()
        self.num_features = num_features
        self.num_domains = num_domains
        self.eps = eps

        self.weight = nn.Parameter(torch.Tensor(num_features))
        self.bias = nn.Parameter(torch.Tensor(num_features))

        self.bn = nn.BatchNorm2d(num_features, eps=eps)

        self.register_buffer("domain_mean", torch.zeros(num_domains, num_features))
        self.register_buffer("domain_var", torch.ones(num_domains, num_features))

        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.ones_(self.weight)
        nn.init.zeros_(self.bias)
        self.bn.reset_parameters()

    def forward(self, x: Tensor, domain_id: int = 0) -> Tensor:
        if self.training:
            return self.bn(x)

        mean = self.domain_mean[domain_id].view(1, -1, 1, 1)
        var = self.domain_var[domain_id].view(1, -1, 1, 1)

        normalized = (x - mean) / (var + self.eps).sqrt()
        return self.weight.view(1, -1, 1, 1) * normalized + self.bias.view(1, -1, 1, 1)


class MultiHeadDomainAttention(Module):
    """Multi-head attention for domain adaptation.

    Args:
        feature_dim: Dimension of input features.
        num_heads: Number of attention heads.
        dropout: Dropout probability.
    """

    def __init__(
        self,
        feature_dim: int,
        num_heads: int = 8,
        dropout: float = 0.1,
    ):
        super().__init__()
        assert feature_dim % num_heads == 0

        self.feature_dim = feature_dim
        self.num_heads = num_heads
        self.head_dim = feature_dim // num_heads
        self.scale = self.head_dim**-0.5

        self.q_proj = nn.Linear(feature_dim, feature_dim)
        self.k_proj = nn.Linear(feature_dim, feature_dim)
        self.v_proj = nn.Linear(feature_dim, feature_dim)

        self.out_proj = nn.Linear(feature_dim, feature_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: Tensor) -> Tensor:
        batch_size = x.size(0)

        Q = (
            self.q_proj(x)
            .view(batch_size, -1, self.num_heads, self.head_dim)
            .transpose(1, 2)
        )
        K = (
            self.k_proj(x)
            .view(batch_size, -1, self.num_heads, self.head_dim)
            .transpose(1, 2)
        )
        V = (
            self.v_proj(x)
            .view(batch_size, -1, self.num_heads, self.head_dim)
            .transpose(1, 2)
        )

        attention = (Q @ K.transpose(-2, -1)) * self.scale
        attention = F.softmax(attention, dim=-1)
        attention = self.dropout(attention)

        out = attention @ V
        out = out.transpose(1, 2).contiguous().view(batch_size, -1, self.feature_dim)

        return self.out_proj(out)


class ProtoNet(Module):
    """Prototypical network for few-shot domain adaptation.

    Args:
        feature_dim: Dimension of input features.
        num_classes: Number of classes.
    """

    def __init__(
        self,
        feature_dim: int,
        num_classes: int,
    ):
        super().__init__()
        self.feature_dim = feature_dim
        self.num_classes = num_classes

        self.encoder = nn.Sequential(
            nn.Linear(feature_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
        )

        self.register_buffer("prototypes", torch.zeros(num_classes, 64))

    def forward(self, x: Tensor) -> Tensor:
        return self.encoder(x)

    def update_prototypes(
        self,
        features: Tensor,
        labels: Tensor,
    ) -> None:
        for c in range(self.num_classes):
            mask = labels == c
            if mask.sum() > 0:
                self.prototypes[c] = features[mask].mean(dim=0)

    def compute_distances(self, features: Tensor) -> Tensor:
        return torch.cdist(features, self.prototypes)


class DomainMixer(Module):
    """Domain mixing module for data augmentation.

    Mixes features from source and target domains.

    Args:
        feature_dim: Dimension of input features.
        mixing_type: Type of mixing ('cross', 'interpolate').
    """

    def __init__(
        self,
        feature_dim: int,
        mixing_type: str = "cross",
    ):
        super().__init__()
        self.feature_dim = feature_dim
        self.mixing_type = mixing_type

        self.mlp = nn.Sequential(
            nn.Linear(feature_dim * 2, feature_dim),
            nn.ReLU(),
            nn.Linear(feature_dim, feature_dim),
        )

    def forward(
        self,
        source: Tensor,
        target: Tensor,
        alpha: float = 0.5,
    ) -> Tensor:
        if self.mixing_type == "interpolate":
            return alpha * source + (1 - alpha) * target
        elif self.mixing_type == "cross":
            mixed = torch.cat([source, target], dim=-1)
            return self.mlp(mixed)
        else:
            raise ValueError(f"Unknown mixing type: {self.mixing_type}")


class UniversalAdapter(Module):
    """Universal domain adapter using bottleneck layer.

    Args:
        feature_dim: Dimension of input features.
        bottleneck_dim: Dimension of bottleneck.
    """

    def __init__(
        self,
        feature_dim: int,
        bottleneck_dim: int = 64,
    ):
        super().__init__()

        self.adapter = nn.Sequential(
            nn.Linear(feature_dim, bottleneck_dim),
            nn.ReLU(),
            nn.Linear(bottleneck_dim, feature_dim),
        )

    def forward(self, x: Tensor) -> Tensor:
        return x + self.adapter(x)


class DomainAttention(Module):
    """Domain attention module for weighted feature aggregation.

    Args:
        feature_dim: Dimension of input features.
        num_domains: Number of domains.
    """

    def __init__(
        self,
        feature_dim: int,
        num_domains: int = 2,
    ):
        super().__init__()

        self.domain_embedding = nn.Embedding(num_domains, feature_dim)
        self.attention = nn.Linear(feature_dim, 1)

    def forward(
        self,
        features: Tensor,
        domain_id: int,
    ) -> Tensor:
        domain_vector = self.domain_embedding(
            torch.tensor(domain_id, device=features.device)
        )

        domain_vector = domain_vector.unsqueeze(0).expand_as(features)

        attention_scores = self.attention(features * domain_vector)
        attention_weights = torch.sigmoid(attention_scores)

        return features * attention_weights
