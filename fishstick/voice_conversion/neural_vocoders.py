"""
Neural Vocoders Module for Voice Conversion

This module provides neural vocoder implementations for converting
spectral representations to waveforms:
- WaveNetVocoder: High-quality autoregressive vocoder
- ParallelWaveGAN: Fast parallel generation vocoder
- HiFiGANVocoder: High-fidelity GAN-based vocoder
"""

from typing import Optional, Dict, Any
from dataclasses import dataclass, field
from abc import ABC, abstractmethod

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


@dataclass
class VocoderConfig:
    """Configuration for neural vocoder."""

    hop_length: int = 256
    sample_rate: int = 22050
    win_length: int = 1024
    n_fft: int = 1024
    n_mels: int = 80
    fmin: float = 0.0
    fmax: float = 8000.0
    model_capacity: str = "medium"
    upsample_rates: list = field(default_factory=lambda: [8, 8, 2, 2])
    upsample_kernel_sizes: list = field(default_factory=lambda: [16, 16, 4, 4])
    resblock_kernel_sizes: list = field(default_factory=lambda: [3, 7, 11])
    resblock_dilation_sizes: list = field(
        default_factory=lambda: [[1, 3, 5], [1, 3, 5], [1, 3, 5]]
    )


class VocoderInterface(ABC):
    """Abstract interface for neural vocoders."""

    @abstractmethod
    def forward(self, mel: Tensor) -> Tensor:
        pass

    @abstractmethod
    def inference(self, mel: Tensor) -> Tensor:
        pass


class ResidualBlock(nn.Module):
    """Residual block for WaveNet/HiFi-GAN vocoder."""

    def __init__(
        self,
        channels: int,
        kernel_size: int = 3,
        dilation_rates: list = [1, 3, 5],
    ):
        super().__init__()
        self.convs = nn.ModuleList()
        for dilation in dilation_rates:
            self.convs.append(
                nn.Conv1d(
                    channels,
                    channels,
                    kernel_size,
                    padding=(kernel_size - 1) * dilation // 2,
                    dilation=dilation,
                )
            )

    def forward(self, x: Tensor) -> Tensor:
        for conv in self.convs:
            x = x + F.leaky_relu(conv(x), 0.2)
        return x


class WaveNetVocoder(nn.Module):
    """WaveNet-based neural vocoder for high-quality waveform generation.

    Autoregressive vocoder that generates samples sequentially.
    Supports both conditional and unconditional generation.

    Args:
        config: VocoderConfig with model parameters
    """

    def __init__(
        self,
        config: VocoderConfig,
        num_classes: int = 256,
        cin_channels: int = 80,
        gin_channels: int = 0,
        skip_channels: int = 512,
        residual_channels: int = 512,
        layers_per_block: int = 10,
    ):
        super().__init__()
        self.config = config
        self.num_classes = num_classes

        self.input_conv = nn.Conv1d(num_classes, residual_channels, 1)

        self.residual_convs = nn.ModuleList()
        self.skip_convs = nn.ModuleList()

        for _ in range(layers_per_block):
            self.residual_convs.append(
                nn.Conv1d(residual_channels, residual_channels, 3, padding=2)
            )
            self.skip_convs.append(nn.Conv1d(residual_channels, skip_channels, 1))

        self.end_conv = nn.Sequential(
            nn.ReLU(),
            nn.Conv1d(skip_channels, skip_channels, 1),
            nn.ReLU(),
            nn.Conv1d(skip_channels, num_classes, 1),
        )

        self.skip_channels = skip_channels
        self.residual_channels = residual_channels
        self.layers_per_block = layers_per_block

    def forward(self, x: Tensor, cond: Optional[Tensor] = None) -> Tensor:
        x = self.input_conv(x)

        skip_sum = torch.zeros(
            x.size(0),
            self.skip_channels,
            x.size(2),
            device=x.device,
        )

        for i in range(self.layers_per_block):
            x = F.relu(x)
            x = F.conv1d(
                x,
                self.residual_convs[i].weight,
                bias=self.residual_convs[i].bias,
                dilation=2**i,
                padding=(2**i * 3 - 1) // 2,
            )
            skip_sum = skip_sum + F.relu(self.skip_convs[i](x))

        x = self.end_conv(skip_sum)
        return x

    def inference(self, mel: Tensor, length: Optional[int] = None) -> Tensor:
        if length is None:
            length = mel.size(2) * self.config.hop_length

        wav = torch.zeros(1, 1, length, device=mel.device, dtype=torch.long)

        for i in range(length):
            if i % self.config.hop_length == 0:
                mel_idx = i // self.config.hop_length
                if mel_idx >= mel.size(2):
                    break
                current_mel = mel[:, :, mel_idx : mel_idx + 1]

            x = F.one_hot(wav[:, :, max(0, i - 100) : i + 1], self.num_classes).float()
            x = x.transpose(1, 2)

            x = self.forward(x, current_mel)
            wav_next = torch.argmax(x[:, :, -1:], dim=1)
            wav[:, :, i + 1 : i + 2] = wav_next

        return wav.squeeze(0)


class ParallelWaveGANGenerator(nn.Module):
    """Parallel WaveGAN generator for fast waveform synthesis.

    Non-autoregressive vocoder using transposed convolutions for
    parallel generation of audio waveforms.

    Args:
        config: VocoderConfig with model parameters
    """

    def __init__(
        self,
        config: VocoderConfig,
        in_channels: int = 80,
        out_channels: int = 1,
        hidden_channels: int = 512,
        num_layers: int = 30,
    ):
        super().__init__()
        self.config = config

        self.input_conv = nn.Conv1d(in_channels, hidden_channels, 1)

        self.upsamples = nn.ModuleList()
        in_ch = hidden_channels
        for rate in config.upsample_rates:
            self.upsamples.append(
                nn.ConvTranspose1d(
                    in_ch,
                    in_ch // 2,
                    rate * 2,
                    stride=rate,
                    padding=rate // 2,
                )
            )
            in_ch = in_ch // 2

        self.resblocks = nn.ModuleList()
        for i in range(num_layers):
            ch = in_ch
            self.resblocks.append(ResidualBlock(ch))

        self.output_conv = nn.Conv1d(in_ch, out_channels, 1)

    def forward(self, mel: Tensor) -> Tensor:
        x = self.input_conv(mel)

        for upsample in self.upsamples:
            x = F.leaky_relu(upsample(x), 0.2)

        for resblock in self.resblocks:
            x = resblock(x)

        x = self.output_conv(x)
        x = torch.tanh(x)
        return x


class ParallelWaveGANDiscriminator(nn.Module):
    """Discriminator for Parallel WaveGAN training."""

    def __init__(
        self,
        in_channels: int = 1,
        out_channels: int = 1,
        kernel_sizes: list = [3, 1],
        strides: list = [1, 1],
    ):
        super().__init__()
        self.convs = nn.ModuleList()
        ch = 16
        for i, (ks, st) in enumerate(zip(kernel_sizes, strides)):
            self.convs.append(
                nn.Conv1d(
                    in_channels if i == 0 else ch,
                    ch * 2 if i < len(kernel_sizes) - 1 else out_channels,
                    ks,
                    stride=st,
                    padding=ks // 2,
                )
            )
            if i < len(kernel_sizes) - 1:
                ch = ch * 2

    def forward(self, x: Tensor) -> list:
        features = []
        for conv in self.convs:
            x = F.leaky_relu(conv(x), 0.2)
            features.append(x)
        return features


class HiFiGANVocoder(nn.Module):
    """HiFi-GAN vocoder for high-fidelity waveform generation.

    Multi-period discriminator and multi-scale discriminator
    architecture for high-quality audio synthesis.

    Args:
        config: VocoderConfig with model parameters
    """

    def __init__(self, config: VocoderConfig):
        super().__init__()
        self.config = config

        num_kernels = len(config.resblock_kernel_sizes)
        num_upsamples = len(config.upsample_rates)

        self.conv_pre = nn.Conv1d(config.n_mels, 512, 7, padding=3)

        self.upsamples = nn.ModuleList()
        for i, (u, k) in enumerate(
            zip(config.upsample_rates, config.upsample_kernel_sizes)
        ):
            self.upsamples.append(
                nn.ConvTranspose1d(
                    512 // (2**i),
                    512 // (2 ** (i + 1)),
                    k,
                    stride=u,
                    padding=(k - u) // 2,
                )
            )

        self.resblocks = nn.ModuleList()
        for i in range(num_upsamples):
            ch = 512 // (2 ** (i + 1))
            for k, d in zip(
                config.resblock_kernel_sizes, config.resblock_dilation_sizes
            ):
                self.resblocks.append(ResidualBlock(ch, k, d))

        self.conv_post = nn.Conv1d(ch, 1, 7, padding=3)

    def forward(self, mel: Tensor) -> Tensor:
        x = self.conv_pre(mel)

        for i, upsample in enumerate(self.upsamples):
            x = F.leaky_relu(x, 0.2)
            x = upsample(x)
            xs = None
            for j in range(
                i * len(self.config.resblock_kernel_sizes),
                (i + 1) * len(self.config.resblock_kernel_sizes),
            ):
                if xs is None:
                    xs = self.resblocks[j](x)
                else:
                    xs = xs + self.resblocks[j](x)
            x = xs / len(self.config.resblock_kernel_sizes)

        x = F.leaky_relu(x)
        x = self.conv_post(x)
        x = torch.tanh(x)
        return x

    def inference(self, mel: Tensor) -> Tensor:
        return self.forward(mel)


class MultiPeriodDiscriminator(nn.Module):
    """Multi-period discriminator for HiFi-GAN."""

    def __init__(self, periods: list = [2, 3, 5, 7, 11]):
        super().__init__()
        self.discriminators = nn.ModuleList(
            [PeriodDiscriminator(period) for period in periods]
        )

    def forward(self, x: Tensor) -> list:
        return [disc(x) for disc in self.discriminators]


class PeriodDiscriminator(nn.Module):
    """Period-based discriminator."""

    def __init__(self, period: int = 2):
        super().__init__()
        self.period = period

        self.convs = nn.ModuleList(
            [
                nn.Conv2d(1, 16, (3, 1), (1, 1), padding=(1, 0)),
                nn.Conv2d(16, 64, (3, 1), (1, 1), padding=(1, 0)),
                nn.Conv2d(64, 256, (3, 1), (1, 1), padding=(1, 0)),
            ]
        )

        self.fc = nn.Linear(256, 1)

    def forward(self, x: Tensor) -> list:
        b, c, t = x.size()

        if t % self.period != 0:
            pad = self.period - (t % self.period)
            x = F.pad(x, (0, pad))
            t = t + pad

        x = x.view(b, c, t // self.period, self.period)

        features = []
        for conv in self.convs:
            x = F.leaky_relu(conv(x), 0.2)
            features.append(x)

        x = x.flatten(1, 2)
        x = self.fc(x)
        return [x, features]


class MultiScaleDiscriminator(nn.Module):
    """Multi-scale discriminator for HiFi-GAN."""

    def __init__(self):
        super().__init__()
        self.discriminators = nn.ModuleList(
            [
                ScaleDiscriminator(),
                ScaleDiscriminator(),
                ScaleDiscriminator(),
            ]
        )
        self.pooling = nn.AvgPool1d(4, 2, padding=2)

    def forward(self, x: Tensor) -> list:
        outputs = []
        for disc in self.discriminators:
            outputs.append(disc(x))
            x = self.pooling(x)
        return outputs


class ScaleDiscriminator(nn.Module):
    """Scale-based discriminator."""

    def __init__(self):
        super().__init__()
        self.convs = nn.ModuleList(
            [
                nn.Conv1d(1, 16, 15, padding=7),
                nn.Conv1d(16, 64, 41, padding=20, groups=4),
                nn.Conv1d(64, 256, 41, padding=20, groups=16),
                nn.Conv1d(256, 512, 41, padding=20, groups=16),
            ]
        )

        self.fc = nn.Linear(512, 1)

    def forward(self, x: Tensor) -> list:
        features = []
        for conv in self.convs:
            x = F.leaky_relu(conv(x), 0.2)
            features.append(x)

        x = x.mean(dim=2)
        x = self.fc(x)
        return [x, features]


def create_vocoder(
    vocoder_type: str = "hifigan",
    config: Optional[VocoderConfig] = None,
) -> nn.Module:
    """Factory function to create vocoder instances.

    Args:
        vocoder_type: Type of vocoder ('wavenet', 'pwg', 'hifigan')
        config: VocoderConfig with parameters

    Returns:
        Initialized vocoder module
    """
    if config is None:
        config = VocoderConfig()

    if vocoder_type.lower() == "wavenet":
        return WaveNetVocoder(config)
    elif vocoder_type.lower() == "pwg" or vocoder_type.lower() == "parallelwavegan":
        return ParallelWaveGANGenerator(config)
    elif vocoder_type.lower() == "hifigan":
        return HiFiGANVocoder(config)
    else:
        raise ValueError(f"Unknown vocoder type: {vocoder_type}")
