"""
Spectral Mapping Module for Voice Conversion

This module provides neural network components for mapping spectral features
between different speaker representations:
- SpectralMappingNetwork: Deep neural network for spectral feature conversion
- FrequencyMasking: Data augmentation technique
- SpectralNormalization: Feature normalization
- MelSpectralMapper: Mel-spectrogram conversion
"""

from typing import Optional, Tuple, List
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


@dataclass
class SpectralConfig:
    """Configuration for spectral mapping network."""

    input_channels: int = 80
    output_channels: int = 80
    hidden_channels: int = 512
    num_layers: int = 6
    kernel_size: int = 5
    dilation_rate: int = 2
    dropout: float = 0.1
    use_causal: bool = False
    use_batch_norm: bool = True
    residual_channels: int = 256
    num_residual_layers: int = 2


class SpectralNormalization(nn.Module):
    """Spectral feature normalization with optional conditioning."""

    def __init__(
        self,
        num_features: int,
        eps: float = 1e-5,
        momentum: float = 0.1,
    ):
        super().__init__()
        self.num_features = num_features
        self.eps = eps
        self.momentum = momentum

        self.weight = nn.Parameter(torch.ones(num_features))
        self.bias = nn.Parameter(torch.zeros(num_features))
        self.register_buffer("running_mean", torch.zeros(num_features))
        self.register_buffer("running_var", torch.ones(num_features))

    def forward(self, x: Tensor, training: bool = False) -> Tensor:
        if training:
            mean = x.mean(dim=(0, 2), keepdim=True)
            var = x.var(dim=(0, 2), keepdim=True, unbiased=True)
            self.running_mean = (
                self.momentum * mean.squeeze() + (1 - self.momentum) * self.running_mean
            )
            self.running_var = (
                self.momentum * var.squeeze() + (1 - self.momentum) * self.running_var
            )
        else:
            mean = self.running_mean.view(1, -1, 1)
            var = self.running_var.view(1, -1, 1)

        x = (x - mean) / torch.sqrt(var + self.eps)
        x = x * self.weight.view(1, -1, 1) + self.bias.view(1, -1, 1)
        return x


class ResidualBlock(nn.Module):
    """Residual block with dilated convolution."""

    def __init__(
        self,
        channels: int,
        kernel_size: int = 3,
        dilation: int = 1,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.conv1 = nn.Conv1d(
            channels,
            channels,
            kernel_size,
            padding=(kernel_size - 1) * dilation // 2,
            dilation=dilation,
        )
        self.conv2 = nn.Conv1d(
            channels,
            channels,
            kernel_size,
            padding=(kernel_size - 1) * dilation // 2,
            dilation=dilation,
        )
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(channels)

    def forward(self, x: Tensor) -> Tensor:
        residual = x
        x = self.conv1(x)
        x = x.relu()
        x = self.dropout(x)
        x = self.conv2(x)
        x = x.relu()
        x = x + residual
        x = x.transpose(1, 2)
        x = self.layer_norm(x)
        x = x.transpose(1, 2)
        return x


class SpectralMappingNetwork(nn.Module):
    """Neural network for mapping spectral features between speakers.

    This network takes source spectral features and converts them to target
    speaker characteristics using a dilated convolutional architecture.

    Args:
        config: SpectralMappingConfig with network parameters
    """

    def __init__(self, config: SpectralConfig):
        super().__init__()
        self.config = config

        self.input_proj = nn.Conv1d(config.input_channels, config.hidden_channels, 1)

        self.residual_blocks = nn.ModuleList()
        for i in range(config.num_residual_layers):
            dilation = config.dilation_rate**i
            self.residual_blocks.append(
                ResidualBlock(
                    config.hidden_channels,
                    config.kernel_size,
                    dilation,
                    config.dropout,
                )
            )

        self.output_proj = nn.Conv1d(config.hidden_channels, config.output_channels, 1)

        if config.use_batch_norm:
            self.input_norm = SpectralNormalization(config.input_channels)
        else:
            self.input_norm = None

    def forward(self, x: Tensor) -> Tensor:
        if self.input_norm is not None:
            x = x.transpose(1, 2)
            x = self.input_norm(x, training=self.training)
            x = x.transpose(1, 2)

        x = self.input_proj(x)

        for block in self.residual_blocks:
            x = block(x)

        x = self.output_proj(x)
        return x


class FrequencyMasking(nn.Module):
    """Frequency domain masking augmentation for robust training.

    Applies frequency masking to spectrograms to improve model robustness.
    """

    def __init__(
        self,
        num_freq_bins: int,
        max_mask_width: int = 15,
        p: float = 0.5,
    ):
        super().__init__()
        self.num_freq_bins = num_freq_bins
        self.max_mask_width = max_mask_width
        self.p = p

    def forward(self, spec: Tensor) -> Tensor:
        if not self.training or torch.rand(1).item() > self.p:
            return spec

        mask_width = torch.randint(1, self.max_mask_width + 1, (1,)).item()
        mask_start = torch.randint(0, self.num_freq_bins - mask_width, (1,)).item()

        spec = spec.clone()
        spec[mask_start : mask_start + mask_width, :] = 0
        return spec


class ChannelWiseAttention(nn.Module):
    """Channel attention for spectral features."""

    def __init__(self, channels: int, reduction: int = 16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.max_pool = nn.AdaptiveMaxPool1d(1)

        self.fc = nn.Sequential(
            nn.Linear(channels, channels // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channels // reduction, channels, bias=False),
        )

    def forward(self, x: Tensor) -> Tensor:
        b, c, _ = x.size()
        avg_out = self.fc(self.avg_pool(x).view(b, c))
        max_out = self.fc(self.max_pool(x).view(b, c))
        attention = (avg_out + max_out).sigmoid()
        return x * attention.view(b, c, 1)


class MelSpectralMapper(nn.Module):
    """Mel-spectrogram to mel-spectrogram conversion network.

    Converts mel-spectrograms from source speaker to target speaker.
    """

    def __init__(
        self,
        num_mels: int = 80,
        hidden_channels: int = 256,
        num_layers: int = 4,
        num_heads: int = 4,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.num_mels = num_mels

        self.input_embedding = nn.Linear(num_mels, hidden_channels)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_channels,
            nhead=num_heads,
            dim_feedforward=hidden_channels * 4,
            dropout=dropout,
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers)

        self.channel_attention = ChannelWiseAttention(hidden_channels)

        self.output_projection = nn.Linear(hidden_channels, num_mels)

    def forward(self, mel: Tensor, mask: Optional[Tensor] = None) -> Tensor:
        x = self.input_embedding(mel.transpose(1, 2))
        x = self.transformer(x, src_key_padding_mask=mask)
        x = x.transpose(1, 2)
        x = self.channel_attention(x)
        x = x.transpose(1, 2)
        x = self.output_projection(x)
        return x.transpose(1, 2)


class ConditionalSpectralMapper(nn.Module):
    """Conditional spectral mapper with speaker embeddings."""

    def __init__(
        self,
        num_mels: int = 80,
        speaker_embedding_dim: int = 256,
        hidden_channels: int = 256,
        num_layers: int = 6,
    ):
        super().__init__()
        self.num_mels = num_mels

        self.speaker_embedding_proj = nn.Linear(speaker_embedding_dim, hidden_channels)

        self.spectral_net = SpectralMappingNetwork(
            SpectralConfig(
                input_channels=num_mels,
                output_channels=num_mels,
                hidden_channels=hidden_channels,
                num_layers=num_layers,
            )
        )

    def forward(self, mel: Tensor, speaker_embedding: Tensor) -> Tensor:
        speaker_emb = self.speaker_embedding_proj(speaker_embedding)
        speaker_emb = speaker_emb.unsqueeze(2)

        converted = self.spectral_net(mel)
        converted = converted + speaker_emb
        return converted
