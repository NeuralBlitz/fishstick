"""
Audio Synthesis

Audio synthesis models and interfaces:
- Vocoder (WaveNet, Griffin-Lim, neural)
- Waveform generation
- Text-to-speech interface
- Speech enhancement
"""

from typing import Optional, List, Dict, Tuple, Union
from dataclasses import dataclass, field

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


@dataclass
class SynthesisConfig:
    """Configuration for audio synthesis."""

    sample_rate: int = 16000
    n_fft: int = 1024
    hop_length: int = 256
    win_length: int = 1024
    n_mels: int = 80
    fmin: float = 0
    fmax: float = 8000
    hidden_dim: int = 512
    num_layers: int = 8
    kernel_size: int = 3


class GriffinLimVocoder:
    """Griffin-Lim vocoder for spectrogram inversion.

    Iteratively estimates the phase of a magnitude spectrogram
    to reconstruct the waveform.
    """

    def __init__(
        self,
        n_fft: int = 1024,
        hop_length: int = 256,
        win_length: int = 1024,
        n_iter: int = 32,
        window: str = "hann",
    ):
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.win_length = win_length
        self.n_iter = n_iter
        self.window = window

    def __call__(self, spec_mag: torch.Tensor) -> torch.Tensor:
        """Convert magnitude spectrogram to waveform.

        Args:
            spec_mag: Magnitude spectrogram (n_fft//2+1, n_frames)

        Returns:
            Reconstructed waveform
        """
        device = spec_mag.device

        spec_angle = torch.randn_like(spec_mag) * 2 * np.pi
        spec_complex = spec_mag * torch.exp(1j * spec_angle)

        for _ in range(self.n_iter):
            waveform = torch.istft(
                spec_complex,
                n_fft=self.n_fft,
                hop_length=self.hop_length,
                win_length=self.win_length,
                window=torch.hann_window(self.win_length).to(device),
            )

            spec_complex = torch.stft(
                waveform,
                n_fft=self.n_fft,
                hop_length=self.hop_length,
                win_length=self.win_length,
                window=torch.hann_window(self.win_length).to(device),
                return_complex=True,
            )

            spec_complex = spec_mag * torch.exp(1j * torch.angle(spec_complex))

        waveform = torch.istft(
            spec_complex,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            win_length=self.win_length,
            window=torch.hann_window(self.win_length).to(device),
        )

        return waveform


class MelGANVocoder(nn.Module):
    """MelGAN vocoder for neural waveform generation.

    Converts mel spectrograms to waveforms using transposed convolutions.
    """

    def __init__(
        self,
        n_mels: int = 80,
        upsample_scales: List[int] = [8, 8, 2, 2],
        resblock_kernel_sizes: List[int] = [3, 7, 11],
        resblock_dilation_sizes: List[List[int]] = [[1, 3, 5], [1, 3, 5], [1, 3, 5]],
        initial_channels: int = 512,
    ):
        super().__init__()

        self.num_kernels = len(resblock_kernel_sizes)
        self.num_upsamples = len(upsample_scales)

        self.conv_pre = nn.Conv1d(n_mels, initial_channels, 7, padding=3)

        self.upsamples = nn.ModuleList()
        self.resblocks = nn.ModuleList()

        channels = initial_channels
        for scale in upsample_scales:
            self.upsamples.append(
                nn.ConvTranspose1d(
                    channels,
                    channels // 2,
                    scale * 2,
                    stride=scale,
                    padding=scale // 2 + scale % 2,
                    output_padding=scale % 2,
                )
            )
            channels = channels // 2

            for k, d in zip(resblock_kernel_sizes, resblock_dilation_sizes):
                self.resblocks.append(ResBlock(channels, k, d))

        self.conv_post = nn.Conv1d(channels, 1, 7, padding=3)

    def forward(self, mel: torch.Tensor) -> torch.Tensor:
        """Generate waveform from mel spectrogram.

        Args:
            mel: Mel spectrogram (batch, n_mels, time)

        Returns:
            Generated waveform (batch, 1, samples)
        """
        x = self.conv_pre(mel)

        for i, upsample in enumerate(self.upsamples):
            x = F.leaky_relu(x, 0.2)
            x = upsample(x)

            xs = None
            for j in range(self.num_kernels):
                if xs is None:
                    xs = self.resblocks[i * self.num_kernels + j](x)
                else:
                    xs = xs + self.resblocks[i * self.num_kernels + j](x)
            x = xs / self.num_kernels

        x = F.leaky_relu(x)
        x = self.conv_post(x)

        x = torch.tanh(x)

        return x.squeeze(1)


class ResBlock(nn.Module):
    """Residual block for MelGAN."""

    def __init__(
        self,
        channels: int,
        kernel_size: int = 3,
        dilation_sizes: List[int] = [1, 3, 5],
    ):
        super().__init__()

        self.convs = nn.ModuleList()
        for d in dilation_sizes:
            self.convs.append(
                nn.Sequential(
                    nn.LeakyReLU(0.2),
                    nn.Conv1d(
                        channels,
                        channels,
                        kernel_size,
                        padding=d * (kernel_size - 1) // 2,
                        dilation=d,
                    ),
                )
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for conv in self.convs:
            x = x + conv(x)
        return x


class WaveNetVocoder(nn.Module):
    """WaveNet vocoder for high-quality waveform generation.

    Autoregressive waveform generation using dilated causal convolutions.
    """

    def __init__(
        self,
        n_mels: int = 80,
        hidden_channels: int = 512,
        num_layers: int = 20,
        kernel_size: int = 3,
        num_quantization_bins: int = 256,
    ):
        super().__init__()

        self.n_mels = n_mels
        self.hidden_channels = hidden_channels
        self.num_layers = num_layers
        self.kernel_size = kernel_size
        self.num_quantization_bins = num_quantization_bins

        self.input_conv = nn.Conv1d(1, hidden_channels, 1)

        self.dilation_convs = nn.ModuleList()
        for i in range(num_layers):
            dilation = 2 ** (i % 10)
            self.dilation_convs.append(
                nn.Conv1d(
                    hidden_channels,
                    hidden_channels,
                    kernel_size,
                    padding=dilation * (kernel_size - 1),
                    dilation=dilation,
                )
            )

        self.mel_conv = nn.Conv1d(n_mels, hidden_channels, 1)

        self.output_conv = nn.Sequential(
            nn.Conv1d(hidden_channels, hidden_channels, 1),
            nn.ReLU(),
            nn.Conv1d(hidden_channels, num_quantization_bins, 1),
        )

        self.causal_conv = CausalConv1d(hidden_channels, hidden_channels, 1)

    def forward(
        self,
        mel: torch.Tensor,
        waveform: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Generate waveform from mel spectrogram.

        Args:
            mel: Mel spectrogram (batch, n_mels, time)
            waveform: Optional input waveform for teacher forcing

        Returns:
            Generated waveform logits
        """
        batch, _, time_steps = mel.shape

        if waveform is None:
            waveform = torch.zeros(batch, 1, time_steps * 256).to(mel.device)

        x = self.input_conv(waveform)

        mel = self.mel_conv(mel)

        for i, dilation_conv in enumerate(self.dilation_convs):
            x = x + dilation_conv(x)[:, :, : x.shape[2]]

            x = x + mel[:, :, : x.shape[2]] * 0.5

            x = self.causal_conv(x)

        x = self.output_conv(x)

        return x


class CausalConv1d(nn.Module):
    """Causal convolution for WaveNet."""

    def __init__(self, in_channels: int, out_channels: int, kernel_size: int):
        super().__init__()

        padding = kernel_size - 1

        self.conv = nn.Conv1d(
            in_channels,
            out_channels,
            kernel_size,
            padding=padding,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv(x)
        return x[:, :, :-1]


class NeuralVocoder(nn.Module):
    """Generic neural vocoder interface.

    Unified interface for different vocoder architectures.
    """

    def __init__(
        self,
        vocoder_type: str = "melgan",
        sample_rate: int = 16000,
        n_mels: int = 80,
        **kwargs,
    ):
        super().__init__()

        self.vocoder_type = vocoder_type
        self.sample_rate = sample_rate
        self.n_mels = n_mels

        if vocoder_type == "melgan":
            self.vocoder = MelGANVocoder(
                n_mels=n_mels,
                upsample_scales=kwargs.get("upsample_scales", [8, 8, 2, 2]),
                initial_channels=kwargs.get("initial_channels", 512),
            )
        elif vocoder_type == "wavenet":
            self.vocoder = WaveNetVocoder(
                n_mels=n_mels,
                hidden_channels=kwargs.get("hidden_channels", 512),
                num_layers=kwargs.get("num_layers", 20),
            )
        else:
            raise ValueError(f"Unknown vocoder type: {vocoder_type}")

    def forward(self, mel: torch.Tensor) -> torch.Tensor:
        """Generate waveform from mel spectrogram.

        Args:
            mel: Mel spectrogram (batch, n_mels, time)

        Returns:
            Generated waveform (batch, samples)
        """
        return self.vocoder(mel)

    def inference(self, mel: torch.Tensor) -> torch.Tensor:
        """Inference mode generation."""
        with torch.no_grad():
            return self.forward(mel)


class SpeechEnhancement(nn.Module):
    """Speech enhancement module.

    Removes noise from speech signals using a U-Net architecture.
    """

    def __init__(
        self,
        n_fft: int = 512,
        hop_length: int = 128,
        win_length: int = 512,
        hidden_channels: int = 32,
    ):
        super().__init__()

        self.n_fft = n_fft
        self.hop_length = hop_length
        self.win_length = win_length

        self.encoder = nn.Sequential(
            nn.Conv2d(1, hidden_channels, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(hidden_channels, hidden_channels, 3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(hidden_channels, hidden_channels * 2, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(hidden_channels * 2, hidden_channels * 2, 3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(hidden_channels * 2, hidden_channels * 4, 3, padding=1),
            nn.ReLU(),
        )

        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(
                hidden_channels * 4,
                hidden_channels * 2,
                3,
                stride=2,
                padding=1,
                output_padding=1,
            ),
            nn.ReLU(),
            nn.ConvTranspose2d(
                hidden_channels * 2,
                hidden_channels,
                3,
                stride=2,
                padding=1,
                output_padding=1,
            ),
            nn.ReLU(),
            nn.Conv2d(hidden_channels, hidden_channels, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(hidden_channels, 1, 1),
            nn.Sigmoid(),
        )

    def forward(self, audio: torch.Tensor) -> torch.Tensor:
        """Enhance speech by removing noise.

        Args:
            audio: Noisy audio waveform (batch, samples)

        Returns:
            Enhanced audio waveform (batch, samples)
        """
        device = audio.device

        window = torch.hann_window(self.win_length).to(device)

        spec = torch.stft(
            audio,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            win_length=self.win_length,
            window=window,
            return_complex=True,
        )

        spec_mag = torch.abs(spec)
        spec_phase = torch.angle(spec)

        spec_mag = spec_mag.unsqueeze(1)

        mask = self.encoder(spec_mag)
        mask = self.decoder(mask)

        mask = mask.squeeze(1)

        enhanced_mag = spec_mag.squeeze(1) * mask

        enhanced_spec = enhanced_mag * torch.exp(1j * spec_phase)

        enhanced_audio = torch.istft(
            enhanced_spec,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            win_length=self.win_length,
            window=window,
        )

        return enhanced_audio


class TTSInterface:
    """Text-to-Speech interface.

    Abstract interface for TTS models.
    """

    def __init__(
        self,
        sample_rate: int = 16000,
        n_mels: int = 80,
    ):
        self.sample_rate = sample_rate
        self.n_mels = n_mels

        self.vocoder = None

    def set_vocoder(self, vocoder: nn.Module):
        """Set the vocoder for waveform generation."""
        self.vocoder = vocoder

    def text_to_mel(self, text: str) -> torch.Tensor:
        """Convert text to mel spectrogram.

        Args:
            text: Input text

        Returns:
            Mel spectrogram
        """
        raise NotImplementedError("Subclass must implement text_to_mel")

    def mel_to_waveform(self, mel: torch.Tensor) -> torch.Tensor:
        """Convert mel spectrogram to waveform.

        Args:
            mel: Mel spectrogram

        Returns:
            Waveform
        """
        if self.vocoder is None:
            griffin_lim = GriffinLimVocoder()
            return griffin_lim(mel)

        return self.vocoder(mel)

    def synthesize(self, text: str) -> torch.Tensor:
        """Synthesize speech from text.

        Args:
            text: Input text

        Returns:
            Generated waveform
        """
        mel = self.text_to_mel(text)

        waveform = self.mel_to_waveform(mel)

        return waveform


class VocoderWrapper:
    """Wrapper for neural vocoders with preprocessing."""

    def __init__(
        self,
        vocoder_type: str = "melgan",
        sample_rate: int = 16000,
        n_fft: int = 1024,
        hop_length: int = 256,
        n_mels: int = 80,
    ):
        self.sample_rate = sample_rate
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.n_mels = n_mels

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.vocoder = NeuralVocoder(
            vocoder_type=vocoder_type,
            sample_rate=sample_rate,
            n_mels=n_mels,
        ).to(self.device)

    def __call__(self, mel: torch.Tensor) -> torch.Tensor:
        """Convert mel spectrogram to waveform.

        Args:
            mel: Mel spectrogram (batch, n_mels, time)

        Returns:
            Generated waveform (batch, samples)
        """
        mel = mel.to(self.device)

        with torch.no_grad():
            waveform = self.vocoder(mel)

        return waveform.cpu()


def create_vocoder(
    vocoder_type: str = "melgan",
    sample_rate: int = 16000,
    n_mels: int = 80,
    **kwargs,
) -> nn.Module:
    """Factory function to create vocoder.

    Args:
        vocoder_type: Type of vocoder ("melgan", "wavenet", "griffin_lim")
        sample_rate: Audio sample rate
        n_mels: Number of mel bins
        **kwargs: Additional arguments

    Returns:
        Vocoder model
    """
    if vocoder_type == "melgan":
        return MelGANVocoder(
            n_mels=n_mels,
            upsample_scales=kwargs.get("upsample_scales", [8, 8, 2, 2]),
            initial_channels=kwargs.get("initial_channels", 512),
        )
    elif vocoder_type == "wavenet":
        return WaveNetVocoder(
            n_mels=n_mels,
            hidden_channels=kwargs.get("hidden_channels", 512),
            num_layers=kwargs.get("num_layers", 20),
        )
    elif vocoder_type == "griffin_lim":
        return GriffinLimVocoder(
            n_fft=kwargs.get("n_fft", 1024),
            hop_length=kwargs.get("hop_length", 256),
            win_length=kwargs.get("win_length", 1024),
            n_iter=kwargs.get("n_iter", 32),
        )
    else:
        raise ValueError(f"Unknown vocoder type: {vocoder_type}")
