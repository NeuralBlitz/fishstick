"""
Music Source Separation

Implementation of music source separation models for separating
different instruments (drums, bass, vocals, other) from music recordings.
"""

from typing import Optional, List, Tuple, Dict
import torch
import torch.nn as nn
import torch.nn.functional as F
from fishstick.audio_separation.base import SeparationModel, SeparationResult, STFT


DEFAULT_MUSIC_SOURCES = ["drums", "bass", "vocals", "other"]


class SourceDecoder(nn.Module):
    """Decoder for reconstructing audio from spectrogram sources."""

    def __init__(
        self,
        n_fft: int = 4096,
        hop_length: int = 1024,
    ):
        super().__init__()
        self.n_fft = n_fft
        self.hop_length = hop_length

    def forward(
        self,
        spectrograms: torch.Tensor,
        original_mix: torch.Tensor,
    ) -> torch.Tensor:
        """Reconstruct audio from spectrogram estimates.

        Args:
            spectrograms: Source spectrograms of shape (n_sources, batch, freq, time)
            original_mix: Original mixture for phase reconstruction

        Returns:
            Reconstructed audio sources
        """
        stft = STFT(self.n_fft, self.hop_length)
        mix_stft = stft(original_mix)

        if mix_stft.dim() == 4:
            mix_stft = mix_stft.squeeze(2)

        phase = torch.angle(mix_stft)

        sources = []
        for i in range(spectrograms.shape[0]):
            source_mag = spectrograms[i]
            source_stft = torch.complex(
                source_mag * torch.cos(phase), source_mag * torch.sin(phase)
            )
            source_wav = stft.inverse(source_stft)
            sources.append(source_wav)

        return torch.stack(sources)


class Demucs(SeparationModel):
    """Demucs: Deep Extractor for Music Sources.

    State-of-the-art music source separation using convolutional networks.

    Reference:
        Demucs: Deep Extractor for Music Sources
    """

    def __init__(
        self,
        n_sources: int = 4,
        sample_rate: int = 44100,
        n_fft: int = 4096,
        hop_length: int = 1024,
        hidden_channels: int = 100,
        num_layers: int = 8,
    ):
        super().__init__(n_sources, sample_rate, n_fft, hop_length)

        self.n_sources = n_sources

        self.encoder = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Conv1d(2, hidden_channels, 8, 4, 4),
                    nn.ReLU(),
                )
                for _ in range(num_layers)
            ]
        )

        self.bottleneck = nn.Sequential(
            nn.Conv1d(hidden_channels, hidden_channels, 3, padding=1),
            nn.ReLU(),
        )

        self.decoder = nn.ModuleList(
            [
                nn.Sequential(
                    nn.ConvTranspose1d(hidden_channels, hidden_channels, 8, 4, 4),
                    nn.ReLU(),
                )
                for _ in range(num_layers)
            ]
        )

        self.final = nn.Conv1d(hidden_channels, n_sources * 2, 1)

        self.source_names = DEFAULT_MUSIC_SOURCES[:n_sources]

    def forward(self, mixture: torch.Tensor) -> SeparationResult:
        """Separate music into individual instrument sources.

        Args:
            mixture: Mixed music audio

        Returns:
            SeparationResult with separated sources
        """
        if mixture.dim() == 2:
            mixture = mixture.unsqueeze(1)

        if mixture.shape[1] == 1:
            mixture = mixture.repeat(1, 2, 1)

        x = mixture

        encoder_outs = []
        for enc in self.encoder:
            x = enc(x)
            encoder_outs.append(x)

        x = self.bottleneck(x)

        for i, dec in enumerate(self.decoder):
            skip_idx = len(encoder_outs) - 1 - i
            x = x + encoder_outs[skip_idx]
            x = dec(x)

        x = self.final(x)

        sources = x.reshape(self.n_sources, 2, x.shape[-1])

        return SeparationResult(
            sources=sources,
            source_names=self.source_names,
        )

    def estimate_sources(self, mixture: torch.Tensor) -> torch.Tensor:
        """Estimate music sources from mixture."""
        result = self.forward(mixture)
        return result.sources


class OpenUnmix(SeparationModel):
    """Open-Unmix for music source separation.

    A widely-used open-source music separation model based on
    bidirectional LSTMs.

    Reference:
        Open-Unmix: A Neural Network for Music Source Separation
    """

    def __init__(
        self,
        n_sources: int = 4,
        sample_rate: int = 44100,
        n_fft: int = 4096,
        hop_length: int = 1024,
        hidden_dim: int = 512,
        nb_channels: int = 2,
    ):
        super().__init__(n_sources, sample_rate, n_fft, hop_length)

        self.nb_channels = nb_channels

        self.stft = STFT(n_fft, hop_length)

        n_bins = n_fft // 2 + 1

        self.input_layer = nn.Sequential(
            nn.Linear(n_bins * nb_channels, hidden_dim),
            nn.ReLU(),
        )

        self.lstm = nn.LSTM(
            hidden_dim,
            hidden_dim,
            num_layers=2,
            batch_first=True,
            bidirectional=True,
            dropout=0.3,
        )

        self.output_layer = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, n_bins * n_sources * nb_channels),
        )

        self.source_names = DEFAULT_MUSIC_SOURCES[:n_sources]

    def forward(self, mixture: torch.Tensor) -> SeparationResult:
        """Separate music into sources using OpenUnmix.

        Args:
            mixture: Mixed music audio

        Returns:
            SeparationResult with separated sources
        """
        mix_stft = self.stft(mixture)

        if mix_stft.dim() == 4:
            mix_stft = mix_stft.squeeze(2)

        mag = torch.abs(mix_stft)

        batch, freq, time = mag.shape

        mag = mag.permute(0, 2, 1)

        x = mag.reshape(batch, time, freq * self.nb_channels)

        x = self.input_layer(x)

        x, _ = self.lstm(x)

        x = self.output_layer(x)

        x = x.reshape(batch, time, self.n_sources, freq * self.nb_channels)

        x = x.permute(2, 0, 1, 3)

        sources = []
        for i in range(self.n_sources):
            source_mag = x[i].reshape(batch, time, freq, self.nb_channels)
            source_mag = source_mag.permute(0, 2, 1, 3)

            source_mag = source_mag * mag.unsqueeze(2)

            source_wav = self._mag_to_wav(source_mag, mix_stft)
            sources.append(source_wav)

        sources = torch.stack(sources)

        return SeparationResult(
            sources=sources,
            source_names=self.source_names,
        )

    def _mag_to_wav(
        self,
        source_mag: torch.Tensor,
        mix_stft: torch.Tensor,
    ) -> torch.Tensor:
        """Convert magnitude spectrogram to waveform."""
        phase = torch.angle(mix_stft)

        source_stft = torch.complex(
            source_mag * torch.cos(phase), source_mag * torch.sin(phase)
        )

        return self.stft.inverse(source_stft)

    def estimate_sources(self, mixture: torch.Tensor) -> torch.Tensor:
        """Estimate music sources from mixture."""
        result = self.forward(mixture)
        return result.sources


class MusicSourceSeparator(nn.Module):
    """Unified interface for music source separation.

    Provides a common interface for various music separation models.
    """

    def __init__(
        self,
        model_type: str = "demucs",
        sources: Optional[List[str]] = None,
        sample_rate: int = 44100,
    ):
        super().__init__()

        self.model_type = model_type
        self.sources = sources or DEFAULT_MUSIC_SOURCES
        self.n_sources = len(self.sources)
        self.sample_rate = sample_rate

        if model_type == "demucs":
            self.model = Demucs(
                n_sources=self.n_sources,
                sample_rate=sample_rate,
            )
        elif model_type == "openunmix":
            self.model = OpenUnmix(
                n_sources=self.n_sources,
                sample_rate=sample_rate,
            )
        else:
            raise ValueError(f"Unknown model type: {model_type}")

    def forward(self, mixture: torch.Tensor) -> SeparationResult:
        """Separate music sources.

        Args:
            mixture: Mixed music audio

        Returns:
            SeparationResult with separated sources
        """
        return self.model(mixture)

    def separate(self, mixture: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Separate and return as dictionary.

        Args:
            mixture: Mixed music audio

        Returns:
            Dictionary mapping source names to audio tensors
        """
        result = self.forward(mixture)

        output = {}
        for i, name in enumerate(result.source_names or self.sources):
            if i < result.sources.shape[0]:
                output[name] = result.sources[i]

        return output


class XUMX(nn.Module):
    """Cross-boundary U-Net for Multitrack Music Separation (X-UMX).

    Reference:
        X-UMX: Cross-boundary U-Net for Multitrack Music Separation
    """

    def __init__(
        self,
        n_sources: int = 4,
        n_fft: int = 2048,
        hop_length: int = 512,
        hidden_dim: int = 256,
    ):
        super().__init__()

        self.n_sources = n_sources
        self.n_fft = n_fft
        self.hop_length = hop_length

        self.stft = STFT(n_fft, hop_length)

        freq_bins = n_fft // 2 + 1

        self.encoder = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1),
            nn.LeakyReLU(0.2),
            nn.Conv2d(32, 64, 3, stride=2, padding=1),
            nn.LeakyReLU(0.2),
            nn.Conv2d(64, 128, 3, stride=2, padding=1),
            nn.LeakyReLU(0.2),
            nn.Conv2d(128, 256, 3, stride=2, padding=1),
            nn.LeakyReLU(0.2),
        )

        self.bottleneck = nn.Sequential(
            nn.Conv2d(256, 256, 3, padding=1),
            nn.LeakyReLU(0.2),
            nn.Conv2d(256, 256, 3, padding=1),
            nn.LeakyReLU(0.2),
        )

        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(256, 128, 4, stride=2, padding=1),
            nn.LeakyReLU(0.2),
            nn.ConvTranspose2d(128, 64, 4, stride=2, padding=1),
            nn.LeakyReLU(0.2),
            nn.ConvTranspose2d(64, 32, 4, stride=2, padding=1),
            nn.LeakyReLU(0.2),
            nn.Conv2d(32, n_sources, 3, padding=1),
        )

    def forward(self, audio: torch.Tensor) -> SeparationResult:
        """Separate music using X-UMX."""
        mix_stft = self.stft(audio)
        if mix_stft.dim() == 4:
            mix_stft = mix_stft.squeeze(2)

        mag = torch.abs(mix_stft)

        mag_input = mag.unsqueeze(1)

        encoded = self.encoder(mag_input)

        bottleneck = self.bottleneck(encoded)

        decoded = self.decoder(bottleneck)

        masks = torch.softmax(decoded, dim=1)

        sources = []
        for i in range(self.n_sources):
            source_mag = mag * masks[:, i : i + 1]
            source_stft = torch.complex(
                source_mag * torch.cos(torch.angle(mix_stft)),
                source_mag * torch.sin(torch.angle(mix_stft)),
            )
            source_wav = self.stft.inverse(source_stft)
            sources.append(source_wav)

        sources = torch.stack(sources)

        return SeparationResult(
            sources=sources,
            source_names=DEFAULT_MUSIC_SOURCES[: self.n_sources],
            masks=masks,
        )


class SourceAdapt(nn.Module):
    """Source Adaptation module for music separation.

    Adapts a pre-trained model to specific music styles.
    """

    def __init__(
        self,
        base_model: nn.Module,
        n_adapt_steps: int = 100,
        lr: float = 1e-4,
    ):
        super().__init__()

        self.base_model = base_model
        self.n_adapt_steps = n_adapt_steps
        self.lr = lr

        for param in base_model.parameters():
            param.requires_grad = False

        self.adaptation_layers = nn.ModuleList(
            [nn.Conv2d(256, 256, 1) for _ in range(3)]
        )

    def adapt(self, mixture: torch.Tensor, references: Dict[str, torch.Tensor]):
        """Adapt model to specific music using reference sources."""
        pass

    def forward(self, mixture: torch.Tensor) -> SeparationResult:
        """Forward pass with adaptation."""
        return self.base_model(mixture)


class WaveformMusicSeparator(nn.Module):
    """Waveform-based music source separator.

    End-to-end waveform music separation without STFT.
    """

    def __init__(
        self,
        n_sources: int = 4,
        channels: int = 64,
        depth: int = 10,
    ):
        super().__init__()

        self.n_sources = n_sources

        self.input_conv = nn.Conv1d(2, channels, 15, padding=7)

        self.encoder_blocks = nn.ModuleList()
        for _ in range(depth):
            block = nn.Sequential(
                nn.Conv1d(channels, channels, 15, padding=7),
                nn.BatchNorm1d(channels),
                nn.LeakyReLU(0.2),
            )
            self.encoder_blocks.append(block)

        self.separator = nn.Sequential(
            nn.Conv1d(channels, channels * n_sources, 1),
            nn.Tanh(),
        )

        self.output_conv = nn.Conv1d(channels, channels, 1)

        self.decoder = nn.ConvTranspose1d(channels, 2, 15, padding=7)

        self.source_names = DEFAULT_MUSIC_SOURCES[:n_sources]

    def forward(self, audio: torch.Tensor) -> SeparationResult:
        """Separate music sources in waveform domain."""
        if audio.dim() == 2:
            audio = audio.unsqueeze(1)

        if audio.shape[1] == 1:
            audio = audio.repeat(1, 2, 1)

        x = self.input_conv(audio)

        for block in self.encoder_blocks:
            x = block(x)
            x = F.avg_pool1d(x, 2) + x

        sep = self.separator(x)
        sep = sep.reshape(sep.shape[0], self.n_sources, -1, sep.shape[-1])

        sources = []
        for i in range(self.n_sources):
            source_x = self.output_conv(sep[:, i])
            source = self.decoder(source_x)
            sources.append(source)

        sources = torch.stack(sources)

        return SeparationResult(
            sources=sources,
            source_names=self.source_names,
        )


class SpleeterEncoder(nn.Module):
    """Encoder based on Spleeter architecture.

    Reference:
        Spleeter: Audio Source Separation with Convolutional Neural Networks
    """

    def __init__(
        self,
        n_fft: int = 4096,
        hop_length: int = 1024,
        n_sources: int = 4,
    ):
        super().__init__()

        self.stft = STFT(n_fft, hop_length)

        freq_bins = n_fft // 2 + 1

        self.encoder = nn.Sequential(
            nn.Conv2d(2, 16, 5, padding=2),
            nn.ReLU(),
            nn.Conv2d(16, 32, 5, padding=2, stride=2),
            nn.ReLU(),
            nn.Conv2d(32, 64, 5, padding=2, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 128, 5, padding=2, stride=2),
            nn.ReLU(),
            nn.Conv2d(128, 256, 5, padding=2, stride=2),
            nn.ReLU(),
        )

        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(256, 128, 4, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(128, 64, 4, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, 4, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(32, n_sources, 5, padding=2),
            nn.Sigmoid(),
        )

        self.source_names = DEFAULT_MUSIC_SOURCES[:n_sources]

    def forward(self, mixture: torch.Tensor) -> SeparationResult:
        """Separate using Spleeter-style architecture."""
        mix_stft = self.stft(mixture)

        if mix_stft.dim() == 4:
            mix_stft = mix_stft.squeeze(2)

        mag = torch.abs(mix_stft)
        phase = torch.angle(mix_stft)

        spec = torch.stack([mag, phase], dim=1)

        encoded = self.encoder(spec)

        masks = self.decoder(encoded)

        sources = []
        for i in range(masks.shape[1]):
            source_mag = mag * masks[:, i : i + 1]
            source_stft = torch.complex(
                source_mag * torch.cos(phase), source_mag * torch.sin(phase)
            )
            source_wav = self.stft.inverse(source_stft)
            sources.append(source_wav)

        sources = torch.stack(sources)

        return SeparationResult(
            sources=sources,
            source_names=self.source_names,
            masks=masks,
        )
