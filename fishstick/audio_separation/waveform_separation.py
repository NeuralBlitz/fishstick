"""
Waveform-based Audio Source Separation

Implementation of end-to-end waveform separation models that operate
directly on the raw audio waveform without STFT transforms.
"""

from typing import Optional, List, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F
from fishstick.audio_separation.base import SeparationModel, SeparationResult


class ConvBlock(nn.Module):
    """1D Convolutional block with residual connection."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        stride: int = 1,
        padding: int = 1,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size, stride, padding)
        self.bn = nn.BatchNorm1d(out_channels)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

        self.skip = (
            nn.Conv1d(in_channels, out_channels, 1)
            if in_channels != out_channels
            else nn.Identity()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = self.skip(x)
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        x = self.dropout(x)
        return x + residual


class WaveUNet(SeparationModel):
    """Wave-U-Net for waveform-based source separation.

    An end-to-end U-Net architecture that operates directly on waveforms,
    using encoder-decoder structure with skip connections.

    Reference:
        Wave-U-Net: A Multi-Scale Neural Network for Audio Source Separation
    """

    def __init__(
        self,
        n_sources: int = 2,
        sample_rate: int = 16000,
        channels: int = 32,
        depth: int = 12,
        kernel_size: int = 15,
        dropout: float = 0.2,
    ):
        super().__init__(n_sources, sample_rate)

        self.channels = channels
        self.depth = depth
        self.kernel_size = kernel_size

        self.input_conv = nn.Conv1d(1, channels, kernel_size, padding=kernel_size // 2)

        self.encoder_blocks = nn.ModuleList()
        in_ch = channels
        for i in range(depth):
            out_ch = min(channels * (2**i), 1024)
            self.encoder_blocks.append(
                nn.ModuleList(
                    [
                        ConvBlock(in_ch, out_ch, kernel_size, dropout=dropout),
                        ConvBlock(out_ch, out_ch, kernel_size, dropout=dropout),
                    ]
                )
            )
            in_ch = out_ch

        self.decoder_blocks = nn.ModuleList()
        for i in range(depth):
            out_ch = max(channels * (2 ** (depth - i - 2)), channels)
            in_ch = in_ch * 2
            self.decoder_blocks.append(
                nn.ModuleList(
                    [
                        ConvBlock(in_ch, out_ch, kernel_size, dropout=dropout),
                        ConvBlock(out_ch, out_ch, kernel_size, dropout=dropout),
                    ]
                )
            )
            in_ch = out_ch

        self.output_conv = nn.Conv1d(channels, n_sources, 1)

    def forward(self, mixture: torch.Tensor) -> SeparationResult:
        """Separate sources using Wave-U-Net.

        Args:
            mixture: Mixed audio of shape (batch, channels, time)

        Returns:
            SeparationResult with separated sources
        """
        if mixture.dim() == 2:
            mixture = mixture.unsqueeze(1)

        x = self.input_conv(mixture)

        encoder_outputs = []
        for enc1, enc2 in self.encoder_blocks:
            x = enc1(x)
            encoder_outputs.append(x)
            x = F.avg_pool1d(x, 2)

        for i, (dec1, dec2) in enumerate(self.decoder_blocks):
            x = F.interpolate(x, scale_factor=2, mode="linear", align_corners=False)
            skip_idx = len(encoder_outputs) - 1 - i
            x = torch.cat([x, encoder_outputs[skip_idx]], dim=1)
            x = dec1(x)
            x = dec2(x)

        sources = self.output_conv(x)
        sources = sources.reshape(self.n_sources, *sources.shape[1:])

        return SeparationResult(sources=sources)


class SudoRM_RF(nn.Module):
    """Sudo Random Matrix Refinement (SudoRM-RF) for waveform separation.

    Uses blocked diagonal structure with multiple random matrices
    for efficient permutation-invariant source separation.

    Reference:
        SudoRM-RF: Efficient Networks for Audio Source Separation
    """

    def __init__(
        self,
        n_sources: int = 2,
        input_channels: int = 64,
        hidden_channels: int = 256,
        n_blocks: int = 8,
        kernel_size: int = 3,
        dilation_factor: int = 2,
    ):
        super().__init__()
        self.n_sources = n_sources

        self.input_conv = nn.Conv1d(
            1, input_channels, kernel_size, padding=kernel_size // 2
        )

        self.blocks = nn.ModuleList()
        for _ in range(n_blocks):
            block = nn.ModuleDict(
                {
                    "depthwise": nn.Conv1d(
                        hidden_channels,
                        hidden_channels,
                        kernel_size,
                        padding=kernel_size // 2 * dilation_factor,
                        dilation=dilation_factor,
                        groups=hidden_channels,
                    ),
                    "pointwise": nn.Conv1d(hidden_channels, hidden_channels, 1),
                    "norm": nn.LayerNorm(hidden_channels),
                }
            )
            self.blocks.append(block)

        self.output_conv = nn.Conv1d(input_channels, n_sources * input_channels, 1)

        self.separator = nn.Sequential(
            nn.Conv1d(n_sources * input_channels, hidden_channels, 1),
            nn.ReLU(),
            nn.Conv1d(hidden_channels, n_sources * input_channels, 1),
        )

        self.input_channels = input_channels

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply SudoRM-RF to separate sources.

        Args:
            x: Input audio of shape (batch, channels, time)

        Returns:
            Separated sources of shape (n_sources, batch, channels, time)
        """
        if x.dim() == 2:
            x = x.unsqueeze(1)

        x = self.input_conv(x)

        for block in self.blocks:
            residual = x
            x = block["depthwise"](x)
            x = block["pointwise"](x)
            x = x.permute(0, 2, 1)
            x = block["norm"](x)
            x = x.permute(0, 2, 1)
            x = F.relu(x + residual)

        x = self.output_conv(x)
        x = self.separator(x)

        batch, channels, time = x.shape
        x = x.reshape(batch, self.n_sources, channels // self.n_sources, time)

        return x


class ConvTasNet(SeparationModel):
    """Convolutional Time-Domain Audio Separation Network (Conv-TasNet).

    A state-of-the-art waveform separation network using 1D dilated
    convolutions and temporal convolutional networks.

    Reference:
        Conv-TasNet: Surpassing Ideal Time-Frequency Magnitude Masking
    """

    def __init__(
        self,
        n_sources: int = 2,
        sample_rate: int = 16000,
        N: int = 512,
        L: int = 16,
        B: int = 128,
        H: int = 512,
        P: int = 3,
        X: int = 8,
        R: int = 4,
    ):
        super().__init__(n_sources, sample_rate, n_fft=L * 2, hop_length=L)

        self.N = N
        self.L = L
        self.B = B
        self.H = H
        self.P = P
        self.X = X
        self.R = R

        self.encoder = nn.Conv1d(1, N, L, stride=L // 2, bias=False)

        self.separator = nn.Sequential(
            nn.Conv1d(N, B, 1),
            *[self._make_tcn_block(B, P, H) for _ in range(R)],
            nn.Conv1d(B, N * n_sources, 1),
        )

        self.decoder = nn.ConvTranspose1d(N, 1, L, stride=L // 2, bias=False)

        self._init_weights()

    def _make_tcn_block(
        self,
        channels: int,
        kernel_size: int,
        hidden_dim: int,
    ) -> nn.Module:
        """Create a TCN block with dilated convolutions."""
        return nn.Sequential(
            nn.Conv1d(channels, hidden_dim, 1),
            nn.PReLU(),
            nn.GroupNorm(1, hidden_dim),
            nn.Conv1d(
                hidden_dim,
                hidden_dim,
                kernel_size,
                padding=(kernel_size - 1) // 2,
            ),
            nn.PReLU(),
            nn.GroupNorm(1, hidden_dim),
        )

    def _init_weights(self):
        """Initialize network weights."""
        nn.init.xavier_uniform_(self.encoder.weight)
        nn.init.xavier_uniform_(self.decoder.weight)

    def forward(self, mixture: torch.Tensor) -> SeparationResult:
        """Separate sources using Conv-TasNet.

        Args:
            mixture: Mixed audio of shape (batch, channels, time)

        Returns:
            SeparationResult with separated sources
        """
        if mixture.dim() == 2:
            mixture = mixture.unsqueeze(1)

        mixture = mixture.mean(dim=1, keepdim=True)

        encoded = self.encoder(mixture)

        mask = self.separator(encoded)
        mask = mask.reshape(mask.shape[0], self.n_sources, self.N, -1)
        mask = F.softmax(mask, dim=1)

        separated = []
        for i in range(self.n_sources):
            source = encoded * mask[:, i]
            source_wav = self.decoder(source)
            separated.append(source_wav)

        sources = torch.stack(separated)

        return SeparationResult(sources=sources, masks=mask)

    def estimate_sources(self, mixture: torch.Tensor) -> torch.Tensor:
        """Estimate sources from mixture."""
        result = self.forward(mixture)
        return result.sources


class DCCRN(SeparationModel):
    """Deep Complex Convolution Reconstructor (DCCRN).

    Complex-valued neural network for speech enhancement and separation
    in the time-frequency domain.

    Reference:
        DCCRN: Deep Complex Convolution Reconstructor
    """

    def __init__(
        self,
        n_sources: int = 2,
        sample_rate: int = 16000,
        n_fft: int = 512,
        hop_length: int = 128,
        hidden_channels: int = 32,
        kernel_size: int = 5,
    ):
        super().__init__(n_sources, sample_rate, n_fft, hop_length)

        self.encoder_real = self._build_encoder(hidden_channels, kernel_size)
        self.encoder_imag = self._build_encoder(hidden_channels, kernel_size)

        self.decoder_real = self._build_decoder(hidden_channels, kernel_size)
        self.decoder_imag = self._build_decoder(hidden_channels, kernel_size)

        self.stft = self._build_stft()

    def _build_encoder(self, channels: int, kernel_size: int) -> nn.Module:
        return nn.Sequential(
            nn.Conv2d(1, channels, kernel_size, padding=kernel_size // 2),
            nn.LeakyReLU(0.2),
            nn.Conv2d(channels, channels * 2, kernel_size, padding=kernel_size // 2),
            nn.LeakyReLU(0.2),
            nn.Conv2d(
                channels * 2, channels * 4, kernel_size, padding=kernel_size // 2
            ),
            nn.LeakyReLU(0.2),
        )

    def _build_decoder(self, channels: int, kernel_size: int) -> nn.Module:
        return nn.Sequential(
            nn.ConvTranspose2d(
                channels * 4, channels * 2, kernel_size, padding=kernel_size // 2
            ),
            nn.LeakyReLU(0.2),
            nn.ConvTranspose2d(
                channels * 2, channels, kernel_size, padding=kernel_size // 2
            ),
            nn.LeakyReLU(0.2),
            nn.ConvTranspose2d(channels, 1, kernel_size, padding=kernel_size // 2),
        )

    def _build_stft(self):
        from fishstick.audio_separation.base import STFT

        return STFT(self.n_fft, self.hop_length)

    def forward(self, mixture: torch.Tensor) -> SeparationResult:
        """Separate sources using DCCRN.

        Args:
            mixture: Mixed audio

        Returns:
            SeparationResult with separated sources
        """
        mix_stft = self.stft(mixture)

        if mix_stft.dim() == 4:
            mix_stft = mix_stft.squeeze(2)

        real = mix_stft.real.unsqueeze(1)
        imag = mix_stft.imag.unsqueeze(1)

        encoded_real = self.encoder_real(real)
        encoded_imag = self.encoder_imag(imag)

        combined = encoded_real + encoded_imag

        decoded_real = self.decoder_real(combined)
        decoded_imag = self.decoder_imag(combined)

        mask_real = torch.tanh(decoded_real)
        mask_imag = torch.tanh(decoded_imag)

        est_real = torch.abs(mix_stft.real) * mask_real.squeeze(1)
        est_imag = torch.abs(mix_stft.imag) * mask_imag.squeeze(1)

        est_stft = torch.complex(est_real, est_imag)

        sources = []
        for i in range(self.n_sources):
            source_wav = self.stft.inverse(est_stft)
            sources.append(source_wav)

        sources = torch.stack(sources)

        return SeparationResult(sources=sources)

    def estimate_sources(self, mixture: torch.Tensor) -> torch.Tensor:
        """Estimate sources from mixture."""
        result = self.forward(mixture)
        return result.sources


class DualPathTransformer(nn.Module):
    """Dual Path Transformer for audio source separation.

    Combines dual-path processing with transformer attention
    for effective separation.
    """

    def __init__(
        self,
        n_sources: int = 2,
        d_model: int = 256,
        n_heads: int = 8,
        n_layers: int = 6,
        chunk_size: int = 100,
        hop_size: int = 50,
    ):
        super().__init__()
        self.n_sources = n_sources
        self.d_model = d_model

        self.input_proj = nn.Conv1d(1, d_model, 1)

        self.intra_transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model,
                n_heads,
                dim_feedforward=d_model * 4,
                dropout=0.1,
                batch_first=True,
            ),
            n_layers // 2,
        )

        self.inter_transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model,
                n_heads,
                dim_feedforward=d_model * 4,
                dropout=0.1,
                batch_first=True,
            ),
            n_layers // 2,
        )

        self.output_proj = nn.Conv1d(d_model, n_sources * d_model, 1)

        self.chunk_size = chunk_size
        self.hop_size = hop_size

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply dual-path transformer separation.

        Args:
            x: Input audio of shape (batch, 1, time)

        Returns:
            Separated sources of shape (n_sources, batch, 1, time)
        """
        if x.dim() == 2:
            x = x.unsqueeze(1)

        x = self.input_proj(x)

        chunks = x.unfold(2, self.chunk_size, self.hop_size)
        n_chunks = chunks.shape[2]

        chunk_list = []
        for i in range(n_chunks):
            chunk = chunks[:, :, i].permute(0, 2, 1)
            chunk = self.intra_transformer(chunk)
            chunk_list.append(chunk)

        chunks_stacked = torch.stack(chunk_list, dim=1)
        chunks_stacked = chunks_stacked.reshape(-1, self.chunk_size, self.d_model)

        chunks_stacked = self.inter_transformer(chunks_stacked)

        output_chunks = []
        for i in range(n_chunks):
            chunk = chunks_stacked[:, i]
            chunk = self.output_proj(chunk.permute(0, 2, 1))
            chunk = chunk.reshape(
                x.shape[0], self.n_sources, self.d_model, self.chunk_size
            )
            output_chunks.append(chunk)

        sources = torch.stack(output_chunks, dim=2)
        sources = sources.reshape(x.shape[0], self.n_sources, self.d_model, -1)

        return sources
