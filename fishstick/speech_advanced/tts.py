import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple
import numpy as np


class TextToMel(nn.Module):
    def __init__(
        self,
        vocab_size: int = 148,
        embed_dim: int = 512,
        encoder_hidden: int = 512,
        encoder_layers: int = 4,
        encoder_heads: int = 4,
        decoder_hidden: int = 1024,
        decoder_layers: int = 4,
        decoder_heads: int = 4,
        mel_channels: int = 80,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.embed_dim = embed_dim
        self.mel_channels = mel_channels

        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.positional_encoding = PositionalEncoding(embed_dim, dropout)

        self.encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=embed_dim,
                nhead=encoder_heads,
                dim_feedforward=encoder_hidden,
                dropout=dropout,
                batch_first=True,
                activation="gelu",
            ),
            num_layers=encoder_layers,
        )

        self.decoder = nn.TransformerDecoder(
            nn.TransformerDecoderLayer(
                d_model=embed_dim,
                nhead=decoder_heads,
                dim_feedforward=decoder_hidden,
                dropout=dropout,
                batch_first=True,
                activation="gelu",
            ),
            num_layers=decoder_layers,
        )

        self.mel_projection = nn.Linear(embed_dim, mel_channels)

    def forward(
        self,
        text: torch.Tensor,
        text_mask: Optional[torch.Tensor] = None,
        target_mel: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        x = self.embedding(text)
        x = self.positional_encoding(x)

        memory = self.encoder(x, src_key_padding_mask=text_mask)

        if target_mel is not None:
            target_mel_embedded = self.positional_encoding(target_mel)
            decoder_output = self.decoder(
                target_mel_embedded,
                memory,
                tgt_key_padding_mask=text_mask,
            )
        else:
            decoder_output = self._generate(memory, text_mask)

        mel_output = self.mel_projection(decoder_output)
        return mel_output, memory

    def _generate(
        self,
        memory: torch.Tensor,
        memory_mask: Optional[torch.Tensor],
        max_len: int = 200,
    ) -> torch.Tensor:
        batch_size = memory.size(0)
        device = memory.device

        decoder_input = torch.zeros(batch_size, 1, self.embed_dim, device=device)

        outputs = []
        for _ in range(max_len):
            decoder_output = self.decoder(decoder_input, memory)

            mel = self.mel_projection(decoder_output)
            outputs.append(mel)

            if self.training:
                break

        if outputs:
            return torch.cat(outputs, dim=1)
        return decoder_input


class AttentionAlignment(nn.Module):
    def __init__(
        self,
        encoder_dim: int = 512,
        decoder_dim: int = 512,
        attention_dim: int = 128,
    ):
        super().__init__()
        self.encoder_dim = encoder_dim
        self.decoder_dim = decoder_dim

        self.encoder_projection = nn.Linear(encoder_dim, attention_dim)
        self.decoder_projection = nn.Linear(decoder_dim, attention_dim)
        self.attention = nn.Linear(attention_dim, 1)

    def forward(
        self,
        encoder_output: torch.Tensor,
        decoder_state: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        encoder_proj = self.encoder_projection(encoder_output)
        decoder_proj = self.decoder_projection(decoder_state)

        decoder_proj = decoder_proj.unsqueeze(1).expand_as(encoder_proj)

        energy = self.attention(torch.tanh(encoder_proj + decoder_proj))
        energy = energy.squeeze(-1)

        if mask is not None:
            energy = energy.masked_fill(mask, float("-inf"))

        attention_weights = F.softmax(energy, dim=-1)

        context = torch.bmm(attention_weights.unsqueeze(1), encoder_output)
        context = context.squeeze(1)

        return context, attention_weights


class GriffinLimVocoder(nn.Module):
    def __init__(
        self,
        n_fft: int = 1024,
        hop_length: int = 256,
        win_length: int = 1024,
        n_mels: int = 80,
        n_iter: int = 32,
    ):
        super().__init__()
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.win_length = win_length
        self.n_mels = n_mels
        self.n_iter = n_iter

        self.register_buffer("window", torch.hann_window(win_length))

    def forward(self, mel_spectrogram: torch.Tensor) -> torch.Tensor:
        mel_to_linear = nn.Linear(self.n_mels, self.n_fft // 2 + 1)
        mel_spec = mel_to_linear(mel_spectrogram)

        spec = self._mel_to_linear(mel_spec)

        waveform = self._griffin_lim(spec)
        return waveform

    def _mel_to_linear(self, mel_spec: torch.Tensor) -> torch.Tensor:
        n_mels = mel_spec.size(-1)
        freq_bins = self.n_fft // 2 + 1

        mel_basis = torch.tensor(
            self._mel_filterbank(freq_bins, self.n_mels),
            device=mel_spec.device,
            dtype=mel_spec.dtype,
        )

        linear_spec = torch.mm(
            torch.pinverse(mel_basis), mel_spec.transpose(1, 2)
        ).transpose(1, 2)

        return linear_spec

    def _griffin_lim(self, spec: torch.Tensor) -> torch.Tensor:
        phase = torch.rand_like(spec)
        complex_spec = torch.cat(
            [spec * torch.cos(phase), spec * torch.sin(phase)], dim=-1
        )

        for _ in range(self.n_iter):
            waveform = torch.istft(
                complex_spec,
                self.n_fft,
                self.hop_length,
                self.win_length,
                window=self.window,
            )

            complex_spec = torch.stft(
                waveform,
                self.n_fft,
                self.hop_length,
                self.win_length,
                window=self.window,
                return_complex=True,
            )

            mag = torch.abs(complex_spec)
            phase = torch.angle(complex_spec)
            complex_spec = torch.cat(
                [mag * torch.cos(phase), mag * torch.sin(phase)], dim=-1
            )

        return waveform

    def _mel_filterbank(self, n_freqs: int, n_mels: int) -> np.ndarray:
        fmin = 0
        fmax = 8000
        f = np.linspace(fmin, fmax, n_freqs + 1)
        hz = f
        mel_min = self._hz_to_mel(fmin)
        mel_max = self._hz_to_mel(fmax)
        mel = np.linspace(mel_min, mel_max, n_mels + 2)
        hz_to_mel = self._hz_to_mel(hz)

        fdiff = np.diff(mel)
        ramps = np.subtract.outer(mel, hz_to_mel)

        lower = -ramps[:-1] / fdiff[1:]
        upper = ramps[1:] / fdiff[:-1]
        filterbank = np.maximum(0, np.minimum(lower, upper))

        return filterbank

    def _hz_to_mel(self, hz: float) -> float:
        return 2595 * np.log10(1 + hz / 700)


class TTSModel(nn.Module):
    def __init__(
        self,
        vocab_size: int = 148,
        embed_dim: int = 512,
        encoder_hidden: int = 512,
        encoder_layers: int = 4,
        encoder_heads: int = 4,
        decoder_hidden: int = 1024,
        decoder_layers: int = 4,
        decoder_heads: int = 4,
        mel_channels: int = 80,
        n_fft: int = 1024,
        hop_length: int = 256,
        win_length: int = 1024,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.mel_channels = mel_channels

        self.text_to_mel = TextToMel(
            vocab_size=vocab_size,
            embed_dim=embed_dim,
            encoder_hidden=encoder_hidden,
            encoder_layers=encoder_layers,
            encoder_heads=encoder_heads,
            decoder_hidden=decoder_hidden,
            decoder_layers=decoder_layers,
            decoder_heads=decoder_heads,
            mel_channels=mel_channels,
            dropout=dropout,
        )

        self.vocoder = GriffinLimVocoder(
            n_fft=n_fft,
            hop_length=hop_length,
            win_length=win_length,
            n_mels=mel_channels,
        )

    def forward(
        self,
        text: torch.Tensor,
        text_mask: Optional[torch.Tensor] = None,
        target_mel: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        mel_output, memory = self.text_to_mel(
            text,
            text_mask,
            target_mel,
        )

        if self.training:
            return mel_output, memory

        waveform = self.vocoder(mel_output)
        return mel_output, waveform

    def infer(
        self,
        text: torch.Tensor,
        text_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        mel_output, _ = self.text_to_mel(text, text_mask, None)
        waveform = self.vocoder(mel_output)
        return mel_output, waveform


class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model)
        )
        pe = torch.zeros(1, max_len, d_model)
        pe[0, :, 0::2] = torch.sin(position * div_term)
        pe[0, :, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.pe[:, : x.size(1), :]
        return self.dropout(x)


class LengthRegulator(nn.Module):
    def __init__(self, encoder_dim: int = 512):
        super().__init__()
        self.encoder_dim = encoder_dim
        self.duration_predictor = nn.Sequential(
            nn.Linear(encoder_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(128, 1),
        )

    def forward(
        self,
        encoder_output: torch.Tensor,
        durations: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if durations is None:
            durations = self.duration_predictor(encoder_output)
            durations = torch.clamp(torch.round(durations.squeeze(-1)), min=1)

        max_len = durations.sum(dim=1).max().int()
        batch_size = encoder_output.size(0)

        expanded = []
        for b in range(batch_size):
            expanded_b = []
            for i, dur in enumerate(durations[b]):
                dur = int(dur.item())
                expanded_b.append(encoder_output[b, i : i + 1].expand(dur, -1))
            expanded.append(torch.cat(expanded_b, dim=0))

        padded = []
        for b in range(batch_size):
            expanded_b = expanded[b]
            if expanded_b.size(0) < max_len:
                padding = torch.zeros(
                    max_len - expanded_b.size(0),
                    self.encoder_dim,
                    device=encoder_output.device,
                )
                expanded_b = torch.cat([expanded_b, padding], dim=0)
            elif expanded_b.size(0) > max_len:
                expanded_b = expanded_b[:max_len]
            padded.append(expanded_b)

        output = torch.stack(padded, dim=0)
        return output, durations


class DurationPredictorLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(
        self,
        pred_durations: torch.Tensor,
        target_durations: torch.Tensor,
    ) -> torch.Tensor:
        loss = F.mse_loss(pred_durations, target_durations.float())
        return loss
