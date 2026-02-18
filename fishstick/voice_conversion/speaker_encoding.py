"""
Speaker Encoding Module for Voice Conversion

This module provides speaker embedding extraction and encoder networks:
- SpeakerEmbeddingNetwork: Encoder for speaker embeddings
- GE2ELoss: Generalized End-to-End loss
- AngularProtoLoss: Angular prototypical loss
- SpeakerEncoderAdvanced: Advanced encoder with attention
"""

from typing import Optional, Tuple, List
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


@dataclass
class SpeakerConfig:
    """Configuration for speaker encoder."""

    input_dim: int = 80
    hidden_dim: int = 256
    embedding_dim: int = 256
    num_layers: int = 3
    dropout: float = 0.1
    attention_heads: int = 4
    use_attention: bool = True
    use_stats: bool = True
    angular_margin: float = 0.2
    scale: float = 30.0


class TDNNBlock(nn.Module):
    """Time Delay Neural Network block for speaker encoding."""

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        kernel_size: int = 5,
        dilation: int = 1,
    ):
        super().__init__()
        self.conv = nn.Conv1d(
            input_dim,
            output_dim,
            kernel_size,
            dilation=dilation,
            padding=(kernel_size - 1) * dilation // 2,
        )
        self.activation = nn.ReLU()
        self.batch_norm = nn.BatchNorm1d(output_dim)
        self.dropout = nn.Dropout(0.1)

    def forward(self, x: Tensor) -> Tensor:
        x = self.conv(x)
        x = self.activation(x)
        x = self.batch_norm(x)
        x = self.dropout(x)
        return x


class SpeakerEmbeddingNetwork(nn.Module):
    """TDNN-based speaker embedding extractor.

    Extracts speaker embeddings from mel-spectrograms using TDNN architecture.
    Suitable for speaker verification and voice conversion.

    Args:
        config: SpeakerConfig with model parameters
    """

    def __init__(self, config: SpeakerConfig):
        super().__init__()
        self.config = config

        self.input_layer = TDNNBlock(
            config.input_dim, config.hidden_dim, kernel_size=5, dilation=1
        )

        self.tdnn_blocks = nn.ModuleList(
            [
                TDNNBlock(
                    config.hidden_dim, config.hidden_dim, kernel_size=3, dilation=2
                ),
                TDNNBlock(
                    config.hidden_dim, config.hidden_dim, kernel_size=3, dilation=3
                ),
                TDNNBlock(
                    config.hidden_dim, config.hidden_dim, kernel_size=1, dilation=1
                ),
            ]
        )

        if config.use_attention:
            self.attention = nn.Sequential(
                nn.Linear(config.hidden_dim, config.hidden_dim),
                nn.ReLU(),
                nn.Linear(config.hidden_dim, 1),
            )

        self.statistics_pooling = nn.Linear(config.hidden_dim, config.hidden_dim * 2)
        self.embedding_layer = nn.Linear(config.hidden_dim * 2, config.embedding_dim)

    def forward(self, x: Tensor, mask: Optional[Tensor] = None) -> Tensor:
        x = x.transpose(1, 2)

        x = self.input_layer(x)

        for block in self.tdnn_blocks:
            x = block(x)

        if hasattr(self, "attention"):
            attn_weights = self.attention(x.transpose(1, 2))
            if mask is not None:
                attn_weights = attn_weights.masked_fill(mask.unsqueeze(-1) == 0, -1e9)
            attn_weights = F.softmax(attn_weights, dim=1)
            x = x * attn_weights.transpose(1, 2)

        if self.config.use_stats:
            mean = x.mean(dim=2)
            std = x.std(dim=2)
            x = torch.cat([mean, std], dim=1)
        else:
            x = x.mean(dim=2)

        x = self.embedding_layer(x)
        return F.normalize(x, p=2, dim=1)


class Res2Block(nn.Module):
    """Res2Net style block for speaker encoding."""

    def __init__(
        self,
        channels: int,
        kernel_size: int = 3,
        scale: int = 8,
    ):
        super().__init__()
        self.scale = scale

        self.convs = nn.ModuleList(
            [
                nn.Conv1d(channels, channels, kernel_size, padding=kernel_size // 2)
                for _ in range(scale - 1)
            ]
        )
        self.alpha = nn.Parameter(torch.ones(scale - 1))

    def forward(self, x: Tensor) -> Tensor:
        spx = x.chunk(self.scale, dim=1)
        outputs = [spx[0]]
        for i in range(1, self.scale):
            if i == 1:
                sp = spx[i]
            else:
                sp = sp + spx[i]
            sp = self.convs[i - 1](sp)
            outputs.append(sp)

        return torch.cat(outputs, dim=1)


class SpeakerEncoderAdvanced(nn.Module):
    """Advanced speaker encoder with Res2Net and attention.

    State-of-the-art speaker encoder using Res2Net blocks and
    multi-head attention for robust embedding extraction.

    Args:
        config: SpeakerConfig with model parameters
    """

    def __init__(self, config: SpeakerConfig):
        super().__init__()
        self.config = config

        self.input_conv = nn.Conv1d(config.input_dim, config.hidden_dim, 5, padding=2)

        self.layer1 = Res2Block(config.hidden_dim, scale=8)
        self.layer2 = Res2Block(config.hidden_dim * 2, scale=8)
        self.layer3 = Res2Block(config.hidden_dim * 4, scale=8)

        self.channel_proj = nn.Sequential(
            nn.Conv1d(config.hidden_dim * 8, config.hidden_dim, 1),
            nn.ReLU(),
        )

        if config.use_attention:
            self.attention = nn.MultiheadAttention(
                config.hidden_dim,
                config.attention_heads,
                dropout=config.dropout,
                batch_first=True,
            )

        self.fc = nn.Linear(config.hidden_dim, config.embedding_dim)

    def forward(self, x: Tensor, mask: Optional[Tensor] = None) -> Tensor:
        x = self.input_conv(x)

        x = self.layer1(x)
        x = F.relu(x)

        x = self.layer2(x)
        x = F.relu(x)

        x = self.layer3(x)
        x = F.relu(x)

        x = self.channel_proj(x)

        if hasattr(self, "attention"):
            x = x.transpose(1, 2)
            x, _ = self.attention(x, x, x, key_padding_mask=mask)
            x = x.transpose(1, 2)

        x = x.mean(dim=2)
        x = self.fc(x)
        return F.normalize(x, p=2, dim=1)


class GE2ELoss(nn.Module):
    """Generalized End-to-End (GE2E) loss for speaker encoding.

    Loss function that directly optimizes speaker embeddings for
    verification tasks.

    Args:
        config: SpeakerConfig with loss parameters
    """

    def __init__(self, config: SpeakerConfig):
        super().__init__()
        self.config = config

    def forward(
        self,
        embeddings: Tensor,
        labels: Tensor,
    ) -> Tensor:
        embeddings = F.normalize(embeddings, p=2, dim=2)

        batch_size = embeddings.size(0)
        num_speakers = labels.unique().size(0)

        centroids = torch.zeros(
            num_speakers, embeddings.size(2), device=embeddings.device
        )

        for i in range(num_speakers):
            mask = labels == i
            centroids[i] = embeddings[mask].mean(dim=0)

        centroids = F.normalize(centroids, p=2, dim=1)

        similarity = torch.matmul(embeddings, centroids.transpose(0, 1))
        similarity = similarity * self.config.scale

        target = labels.unsqueeze(1).expand_as(similarity)
        mask = torch.zeros_like(similarity).scatter_(1, target, 1)

        pos_sim = (similarity * mask).sum(dim=1)
        neg_sim = (similarity * (1 - mask)).max(dim=1)[0]

        loss = F.relu(neg_sim - pos_sim + self.config.angular_margin).mean()
        return loss


class AngularProtoLoss(nn.Module):
    """Angular prototypical loss for speaker encoding.

    Margin-based loss with angular constraints for improved
    speaker discrimination.

    Args:
        config: SpeakerConfig with loss parameters
    """

    def __init__(self, config: SpeakerConfig):
        super().__init__()
        self.config = config
        self.margin = config.angular_margin

    def forward(
        self,
        embeddings: Tensor,
        labels: Tensor,
    ) -> Tensor:
        embeddings = F.normalize(embeddings, p=2, dim=1)

        batch_size = embeddings.size(0)
        num_speakers = labels.max().item() + 1

        prototypes = torch.zeros(
            num_speakers, embeddings.size(1), device=embeddings.device
        )

        for i in range(num_speakers):
            mask = labels == i
            if mask.sum() > 0:
                prototypes[i] = embeddings[mask].mean(dim=0)

        prototypes = F.normalize(prototypes, p=2, dim=1)

        cos_sim = torch.matmul(embeddings, prototypes.t())

        target = labels
        positive_sim = cos_sim.gather(1, target.unsqueeze(1)).squeeze(1)

        cos_sim_m = (
            cos_sim - torch.eye(num_speakers, device=cos_sim.device) * self.margin
        )
        negative_sim = cos_sim_m.max(dim=1)[0]

        loss = F.relu(negative_sim - positive_sim + self.margin).mean()
        return loss


class AAMSoftmax(nn.Module):
    """Additive Angular Margin Softmax for speaker recognition.

    ArcFace-style loss with angular margins for enhanced
    speaker discrimination.

    Args:
        config: SpeakerConfig with loss parameters
    """

    def __init__(
        self,
        embedding_dim: int,
        num_classes: int,
        margin: float = 0.2,
        scale: float = 30.0,
    ):
        super().__init__()
        self.margin = margin
        self.scale = scale

        self.weight = nn.Parameter(torch.FloatTensor(num_classes, embedding_dim))
        nn.init.xavier_uniform_(self.weight)

    def forward(self, embeddings: Tensor, labels: Tensor) -> Tensor:
        normalized_weights = F.normalize(self.weight, p=2, dim=1)

        cos_theta = F.linear(embeddings, normalized_weights)
        cos_theta = cos_theta.clamp(-1, 1)

        theta = torch.acos(cos_theta)
        target_theta = torch.acos(cos_theta.gather(1, labels.unsqueeze(1)).squeeze(1))

        marginal_theta = target_theta + self.margin
        marginal_cos_theta = torch.cos(marginal_theta)

        cos_theta.scatter_(1, labels.unsqueeze(1), marginal_cos_theta)
        cos_theta = cos_theta * self.scale

        loss = F.cross_entropy(cos_theta, labels)
        return loss


class SpeakerEncoderWithLoss(nn.Module):
    """Speaker encoder with integrated loss computation."""

    def __init__(self, config: SpeakerConfig, num_speakers: Optional[int] = None):
        super().__init__()
        self.config = config

        self.encoder = SpeakerEmbeddingNetwork(config)

        if num_speakers is not None:
            self.loss = AAMSoftmax(
                config.embedding_dim,
                num_speakers,
                config.angular_margin,
                config.scale,
            )
        else:
            self.loss = GE2ELoss(config)

    def forward(
        self,
        x: Tensor,
        labels: Optional[Tensor] = None,
        mask: Optional[Tensor] = None,
    ) -> Tuple[Tensor, Optional[Tensor]]:
        embeddings = self.encoder(x, mask)

        if labels is not None and self.training:
            loss = self.loss(embeddings, labels)
            return embeddings, loss

        return embeddings, None


def create_speaker_encoder(
    config: Optional[SpeakerConfig] = None,
    encoder_type: str = "tdnn",
    num_speakers: Optional[int] = None,
) -> nn.Module:
    """Factory function to create speaker encoders.

    Args:
        config: SpeakerConfig with parameters
        encoder_type: Type of encoder ('tdnn', 'res2net', 'advanced')
        num_speakers: Number of speakers for classification

    Returns:
        Initialized speaker encoder
    """
    if config is None:
        config = SpeakerConfig()

    if encoder_type == "tdnn":
        return SpeakerEmbeddingNetwork(config)
    elif encoder_type == "res2net":
        return SpeakerEncoderAdvanced(config)
    elif encoder_type == "advanced":
        return SpeakerEncoderWithLoss(config, num_speakers)
    else:
        raise ValueError(f"Unknown encoder type: {encoder_type}")
