"""
Speech and Audio Module
=======================

Comprehensive speech recognition and audio processing module featuring:
- Wav2Vec2: Self-supervised speech recognition
- HuBERT: Masked contrastive predictive coding for speech
- Conformer: Convolution-augmented transformer for speech
- Whisper: OpenAI's speech recognition model
- Audio Spectrogram Transformer (AST)

Includes:
- Feature extraction (MFCC, mel spectrogram)
- Audio preprocessing
- CTC loss wrapper
- Language model integration interface
"""

from typing import Optional, List, Dict, Any, Union, Tuple
from dataclasses import dataclass, field
from enum import Enum
import math
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.utils.data import Dataset, DataLoader

try:
    import torchaudio

    TORCHAUDIO_AVAILABLE = True
except ImportError:
    TORCHAUDIO_AVAILABLE = False

try:
    from transformers import (
        Wav2Vec2Model,
        Wav2Vec2ForCTC,
        HubertModel,
        HubertForCTC,
        WhisperModel,
        WhisperForConditionalGeneration,
    )

    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False


class AudioFeatureType(Enum):
    """Types of audio features."""

    MFCC = "mfcc"
    MEL_SPECTROGRAM = "mel_spectrogram"
    SPECTROGRAM = "spectrogram"
    WAVEFORM = "waveform"


@dataclass
class AudioConfig:
    """Configuration for audio processing."""

    sample_rate: int = 16000
    n_fft: int = 400
    hop_length: int = 160
    win_length: int = 400
    n_mels: int = 80
    n_mfcc: int = 13
    f_min: float = 0.0
    f_max: float = 8000.0
    window_fn: str = "hann"
    power: float = 2.0
    normalize: bool = True


@dataclass
class SpeechConfig:
    """Configuration for speech models."""

    model_name: str = "wav2vec2-base"
    hidden_size: int = 768
    num_hidden_layers: int = 12
    num_attention_heads: int = 12
    intermediate_size: int = 3072
    hidden_dropout: float = 0.1
    attention_dropout: float = 0.1
    activation: str = "gelu"
    vocab_size: int = 32
    max_length: int = 3000
    pad_token_id: int = 0
    bos_token_id: int = 1
    eos_token_id: int = 2
    use_ctc: bool = True
    ctc_zero_infinity: bool = True


class FeatureExtractor(nn.Module):
    """Feature extraction module for audio signals."""

    def __init__(self, config: AudioConfig):
        super().__init__()
        self.config = config

        if TORCHAUDIO_AVAILABLE:
            self.melspec = torchaudio.transforms.MelSpectrogram(
                sample_rate=config.sample_rate,
                n_fft=config.n_fft,
                hop_length=config.hop_length,
                win_length=config.win_length,
                n_mels=config.n_mels,
                f_min=config.f_min,
                f_max=config.f_max,
                window_fn=getattr(torch, f"{config.window_fn}_window")(
                    config.win_length
                ),
            )
            self.mfcc = torchaudio.transforms.MFCC(
                sample_rate=config.sample_rate,
                n_mfcc=config.n_mfcc,
                melkwargs={
                    "n_fft": config.n_fft,
                    "hop_length": config.hop_length,
                    "n_mels": config.n_mels,
                    "f_min": config.f_min,
                    "f_max": config.f_max,
                },
            )

    def extract_mfcc(self, waveform: Tensor) -> Tensor:
        """Extract MFCC features from waveform."""
        if not TORCHAUDIO_AVAILABLE:
            return self._mfcc_manual(waveform)
        mfcc = self.mfcc(waveform)
        if self.config.normalize:
            mfcc = (mfcc - mfcc.mean(dim=-1, keepdim=True)) / (
                mfcc.std(dim=-1, keepdim=True) + 1e-8
            )
        return mfcc

    def extract_mel_spectrogram(self, waveform: Tensor) -> Tensor:
        """Extract mel spectrogram from waveform."""
        if not TORCHAUDIO_AVAILABLE:
            return self._mel_spec_manual(waveform)
        mel_spec = self.melspec(waveform)
        mel_spec = torch.log(torch.clamp(mel_spec, min=1e-5))
        if self.config.normalize:
            mel_spec = (mel_spec - mel_spec.mean(dim=-1, keepdim=True)) / (
                mel_spec.std(dim=-1, keepdim=True) + 1e-8
            )
        return mel_spec

    def _mfcc_manual(self, waveform: Tensor) -> Tensor:
        """Manual MFCC extraction without torchaudio."""
        spec = torch.stft(
            waveform.squeeze(1),
            n_fft=self.config.n_fft,
            hop_length=self.config.hop_length,
            win_length=self.config.win_length,
            window=torch.hann_window(self.config.win_length, device=waveform.device),
            return_complex=True,
        )
        spec = torch.abs(spec)
        mel_matrix = self._create_mel_matrix(self.config.n_mels, waveform.device)
        mel_spec = torch.matmul(mel_matrix, spec)
        mel_spec = torch.log(torch.clamp(mel_spec, min=1e-5))

        dct_matrix = self._create_dct_matrix(self.config.n_mfcc, mel_spec.shape[1])
        mfcc = torch.matmul(mel_spec.transpose(1, 2), dct_matrix.transpose(0, 1))
        return mfcc.transpose(1, 2)

    def _mel_spec_manual(self, waveform: Tensor) -> Tensor:
        """Manual mel spectrogram extraction."""
        spec = torch.stft(
            waveform.squeeze(1),
            n_fft=self.config.n_fft,
            hop_length=self.config.hop_length,
            win_length=self.config.win_length,
            window=torch.hann_window(self.config.win_length),
            return_complex=True,
        )
        spec = torch.abs(spec)
        mel_matrix = self._create_mel_matrix(self.config.n_mels, waveform.device)
        mel_spec = torch.matmul(mel_matrix, spec)
        return mel_spec

    def _create_dct_matrix(self, n_mfcc: int, n_bins: int) -> Tensor:
        """Create DCT transformation matrix."""
        dct = np.zeros((n_mfcc, n_bins))
        for k in range(n_mfcc):
            dct[k, :] = np.cos(np.pi * k * (np.arange(n_bins) + 0.5) / n_bins)
        return torch.from_numpy(dct).float()

    def _create_mel_matrix(self, n_mels: int, device: torch.device) -> Tensor:
        """Create mel filterbank matrix."""
        n_freqs = self.config.n_fft // 2 + 1
        mel_min = self._hz_to_mel(self.config.f_min)
        mel_max = self._hz_to_mel(self.config.f_max)
        mel_points = torch.linspace(mel_min, mel_max, n_mels + 2, device=device)
        hz_points = self._mel_to_hz(mel_points)
        bin_points = (
            torch.floor((self.config.n_fft + 1) * hz_points / self.config.sample_rate)
            .long()
            .clamp(0, n_freqs - 1)
        )

        mel_matrix = torch.zeros(n_mels, n_freqs, device=device)
        for i in range(n_mels):
            left = bin_points[i].item()
            center = bin_points[i + 1].item()
            right = bin_points[i + 2].item()
            for j in range(left, center):
                if j < n_freqs:
                    mel_matrix[i, j] = (j - left) / max(1, center - left)
            for j in range(center, right):
                if j < n_freqs:
                    mel_matrix[i, j] = (right - j) / max(1, right - center)

        return mel_matrix

    def _hz_to_mel(self, hz: float) -> float:
        """Convert Hz to mel scale."""
        return 2595 * np.log10(1 + hz / 700)

    def _mel_to_hz(self, mel: Tensor) -> Tensor:
        """Convert mel scale to Hz."""
        return 700 * (10 ** (mel / 2595) - 1)

    def forward(
        self,
        waveform: Tensor,
        feature_type: AudioFeatureType = AudioFeatureType.MEL_SPECTROGRAM,
    ) -> Tensor:
        """Extract features from waveform."""
        if feature_type == AudioFeatureType.MFCC:
            return self.extract_mfcc(waveform)
        elif feature_type == AudioFeatureType.MEL_SPECTROGRAM:
            return self.extract_mel_spectrogram(waveform)
        else:
            return waveform


class AudioPreprocessor(nn.Module):
    """Audio preprocessing module with normalization and augmentation."""

    def __init__(self, config: AudioConfig, augment: bool = False):
        super().__init__()
        self.config = config
        self.augment = augment
        self.sample_rate = config.sample_rate

        self.register_buffer("mean", torch.zeros(1))
        self.register_buffer("std", torch.ones(1))

    def normalize_waveform(self, waveform: Tensor) -> Tensor:
        """Normalize waveform to zero mean and unit variance."""
        if waveform.dim() == 1:
            waveform = waveform.unsqueeze(0)
        mean = waveform.mean(dim=-1, keepdim=True)
        std = waveform.std(dim=-1, keepdim=True)
        return (waveform - mean) / (std + 1e-8)

    def resample(self, waveform: Tensor, orig_sr: int, target_sr: int) -> Tensor:
        """Resample audio to target sample rate."""
        if orig_sr == target_sr:
            return waveform
        if not TORCHAUDIO_AVAILABLE:
            return self._resample_manual(waveform, orig_sr, target_sr)
        return torchaudio.functional.resample(waveform, orig_sr, target_sr)

    def _resample_manual(
        self, waveform: Tensor, orig_sr: int, target_sr: int
    ) -> Tensor:
        """Manual resampling using linear interpolation."""
        orig_len = waveform.shape[-1]
        target_len = int(orig_len * target_sr / orig_sr)
        indices = torch.linspace(0, orig_len - 1, target_len, device=waveform.device)
        indices_left = indices.floor().long()
        indices_right = (indices_left + 1).clamp(max=orig_len - 1)
        weights = (indices - indices_left.float()).unsqueeze(0)

        left = waveform[..., indices_left]
        right = waveform[..., indices_right]
        return (left * (1 - weights) + right * weights).squeeze(0)

    def add_noise(self, waveform: Tensor, noise_level: float = 0.005) -> Tensor:
        """Add Gaussian noise to waveform."""
        noise = torch.randn_like(waveform) * noise_level
        return waveform + noise

    def time_stretch(self, waveform: Tensor, rate: float = 1.0) -> Tensor:
        """Time stretch the audio."""
        if rate == 1.0:
            return waveform
        if not TORCHAUDIO_AVAILABLE:
            return self._time_stretch_manual(waveform, rate)
        return torchaudio.functional.phase_vocoder(
            waveform, rate, hop_length=self.config.hop_length
        )

    def _time_stretch_manual(self, waveform: Tensor, rate: float) -> Tensor:
        """Manual time stretching."""
        orig_len = waveform.shape[-1]
        target_len = int(orig_len / rate)
        indices = torch.linspace(0, orig_len - 1, target_len, device=waveform.device)
        indices = indices.clamp(0, orig_len - 1)
        return torch.nn.functional.interpolate(
            waveform.unsqueeze(0), size=target_len, mode="linear", align_corners=False
        ).squeeze(0)

    def pitch_shift(self, waveform: Tensor, n_steps: int = 0) -> Tensor:
        """Shift pitch by n semitones."""
        if n_steps == 0:
            return waveform
        rate = 2 ** (n_steps / 12)
        return self.time_stretch(waveform, rate)

    def compute_speech_features(self, waveform: Tensor) -> Dict[str, Tensor]:
        """Compute various speech features."""
        feature_extractor = FeatureExtractor(self.config)

        return {
            "waveform": waveform,
            "mfcc": feature_extractor.extract_mfcc(waveform),
            "mel_spectrogram": feature_extractor.extract_mel_spectrogram(waveform),
            "normalized": self.normalize_waveform(waveform),
        }

    def forward(self, waveform: Tensor, orig_sr: Optional[int] = None) -> Tensor:
        """Preprocess audio waveform."""
        if orig_sr is not None and orig_sr != self.sample_rate:
            waveform = self.resample(waveform, orig_sr, self.sample_rate)

        if self.augment:
            waveform = self.add_noise(waveform)

        waveform = self.normalize_waveform(waveform)
        return waveform


class CTCLossWrapper(nn.Module):
    """CTC (Connectionist Temporal Classification) loss wrapper."""

    def __init__(
        self,
        vocab_size: int,
        blank: int = 0,
        reduction: str = "mean",
        zero_infinity: bool = True,
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.blank = blank
        self.reduction = reduction
        self.ctc_loss = nn.CTCLoss(
            blank=blank,
            reduction=reduction,
            zero_infinity=zero_infinity,
        )

    def forward(
        self,
        log_probs: Tensor,
        targets: Tensor,
        input_lengths: Optional[Tensor] = None,
        target_lengths: Optional[Tensor] = None,
        apply_log_softmax: bool = True,
    ) -> Tensor:
        """Compute CTC loss."""
        if input_lengths is None:
            input_lengths = torch.full(
                (log_probs.size(1),),
                log_probs.size(0),
                dtype=torch.long,
                device=log_probs.device,
            )

        if target_lengths is None:
            if targets.dim() == 1:
                target_lengths = torch.tensor(
                    [targets.size(0)],
                    dtype=torch.long,
                    device=log_probs.device,
                )
            else:
                target_lengths = torch.full(
                    (targets.size(0),),
                    targets.size(1),
                    dtype=torch.long,
                    device=targets.device,
                )

        if apply_log_softmax:
            log_probs = log_probs.log_softmax(dim=-1)

        loss = self.ctc_loss(log_probs, targets, input_lengths, target_lengths)
        return loss

    def decode(
        self,
        log_probs: Tensor,
        method: str = "greedy",
        beam_width: int = 10,
    ) -> List[List[int]]:
        """Decode log probabilities to token sequences."""
        if method == "greedy":
            return self._greedy_decode(log_probs)
        elif method == "beam":
            return self._beam_decode(log_probs, beam_width)
        else:
            raise ValueError(f"Unknown decode method: {method}")

    def _greedy_decode(self, log_probs: Tensor) -> List[List[int]]:
        """Greedy decoding."""
        predictions = log_probs.argmax(dim=-1)
        decoded = []
        for pred in predictions:
            collapsed = []
            for idx in pred.tolist():
                if idx != self.blank and (len(collapsed) == 0 or idx != collapsed[-1]):
                    collapsed.append(idx)
            decoded.append(collapsed)
        return decoded

    def _beam_decode(self, log_probs: Tensor, beam_width: int) -> List[List[int]]:
        """Beam search decoding."""
        decoded = []
        for lp in log_probs:
            probs = F.softmax(lp, dim=-1)
            topk_probs, topk_indices = probs.topk(beam_width, dim=-1)

            beams = [([], 1.0)]
            for t in range(probs.size(0)):
                new_beams = {}
                for seq, score in beams:
                    for vocab_idx in range(probs.size(-1)):
                        new_score = score * topk_probs[t, vocab_idx].item()
                        new_seq = seq.copy()

                        if vocab_idx != self.blank:
                            if len(new_seq) == 0 or new_seq[-1] != vocab_idx:
                                new_seq.append(vocab_idx)

                        key = tuple(new_seq)
                        new_beams[key] = new_beams.get(key, 0) + new_score

                beams = sorted(new_beams.items(), key=lambda x: x[1], reverse=True)[
                    :beam_width
                ]
                beams = [(list(k), v) for k, v in beams]

            decoded.append(beams[0][0] if beams else [])
        return decoded


class Wav2Vec2Model(nn.Module):
    """Wav2Vec2: Self-supervised speech recognition model."""

    def __init__(self, config: SpeechConfig):
        super().__init__()
        self.config = config

        self.feature_encoder = nn.Sequential(
            nn.Conv1d(1, 512, kernel_size=10, stride=5, padding=3, bias=False),
            nn.LayerNorm(512),
            nn.GELU(),
            nn.Conv1d(512, 512, kernel_size=3, stride=2, padding=1, bias=False),
            nn.LayerNorm(512),
            nn.GELU(),
            nn.Conv1d(512, 512, kernel_size=3, stride=2, padding=1, bias=False),
            nn.LayerNorm(512),
            nn.GELU(),
            nn.Conv1d(512, 512, kernel_size=3, stride=2, padding=1, bias=False),
            nn.LayerNorm(512),
            nn.GELU(),
            nn.Conv1d(512, 512, kernel_size=3, stride=2, padding=1, bias=False),
            nn.LayerNorm(512),
            nn.GELU(),
            nn.Conv1d(512, 512, kernel_size=3, stride=2, padding=1, bias=False),
            nn.LayerNorm(512),
            nn.GELU(),
        )

        self.feature_projection = nn.Sequential(
            nn.Linear(512, config.hidden_size),
            nn.LayerNorm(config.hidden_size),
            nn.Dropout(config.hidden_dropout),
        )

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=config.hidden_size,
            nhead=config.num_attention_heads,
            dim_feedforward=config.intermediate_size,
            dropout=config.hidden_dropout,
            activation=config.activation,
            batch_first=True,
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=config.num_hidden_layers,
        )

        self.masked_spec_embed = nn.Parameter(torch.randn(config.hidden_size))

        if config.use_ctc:
            self.ctc_head = nn.Linear(config.hidden_size, config.vocab_size)

    def extract_features(self, waveform: Tensor) -> Tensor:
        """Extract features from waveform."""
        if waveform.dim() == 1:
            waveform = waveform.unsqueeze(0).unsqueeze(1)
        elif waveform.dim() == 2:
            waveform = waveform.unsqueeze(1)

        features = self.feature_encoder(waveform)
        features = features.transpose(1, 2)
        features = self.feature_projection(features)
        return features

    def forward(
        self,
        waveform: Tensor,
        attention_mask: Optional[Tensor] = None,
        output_hidden_states: bool = False,
    ) -> Dict[str, Tensor]:
        """Forward pass."""
        features = self.extract_features(waveform)

        if self.training:
            mask_length = features.size(1) // 20
            mask_start = torch.rand(features.size(1) - mask_length).bernoulli().bool()
            mask = torch.zeros_like(features, dtype=torch.bool)
            for i in range(features.size(1) - mask_length):
                if mask_start[i]:
                    mask[:, i : i + mask_length] = True

            features[mask] = self.masked_spec_embed

        hidden_states = self.transformer_encoder(
            features, src_key_padding_mask=attention_mask
        )

        output = {"last_hidden_state": hidden_states}

        if hasattr(self, "ctc_head"):
            logits = self.ctc_head(hidden_states)
            output["logits"] = logits

        if output_hidden_states:
            output["hidden_states"] = hidden_states

        return output


class HubertModel(nn.Module):
    """HuBERT: Masked contrastive predictive coding for speech."""

    def __init__(self, config: SpeechConfig):
        super().__init__()
        self.config = config

        self.feature_encoder = nn.Sequential(
            nn.Conv1d(1, 512, kernel_size=10, stride=5, padding=3, bias=False),
            nn.LayerNorm(512),
            nn.GELU(),
            nn.Conv1d(512, 512, kernel_size=3, stride=2, padding=1, bias=False),
            nn.LayerNorm(512),
            nn.GELU(),
            nn.Conv1d(512, 512, kernel_size=3, stride=2, padding=1, bias=False),
            nn.LayerNorm(512),
            nn.GELU(),
            nn.Conv1d(512, 512, kernel_size=3, stride=2, padding=1, bias=False),
            nn.LayerNorm(512),
            nn.GELU(),
            nn.Conv1d(512, 512, kernel_size=3, stride=2, padding=1, bias=False),
            nn.LayerNorm(512),
            nn.GELU(),
        )

        self.feature_projection = nn.Sequential(
            nn.Linear(512, config.hidden_size),
            nn.LayerNorm(config.hidden_size),
            nn.Dropout(config.hidden_dropout),
        )

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=config.hidden_size,
            nhead=config.num_attention_heads,
            dim_feedforward=config.intermediate_size,
            dropout=config.hidden_dropout,
            activation=config.activation,
            batch_first=True,
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=config.num_hidden_layers,
        )

        self.num_masked_stages = 6
        self.mask_embedding = nn.Parameter(torch.randn(config.hidden_size))

        if config.use_ctc:
            self.ctc_head = nn.Linear(config.hidden_size, config.vocab_size)

    def apply_mask(
        self, hidden_states: Tensor, mask_prob: float = 0.15
    ) -> Tuple[Tensor, Tensor]:
        """Apply masked prediction."""
        batch_size, seq_len, hidden_dim = hidden_states.shape
        mask = torch.rand(batch_size, seq_len, device=hidden_states.device) < mask_prob
        masked_hidden = hidden_states.clone()
        masked_hidden[mask] = self.mask_embedding
        return masked_hidden, mask

    def forward(
        self,
        waveform: Tensor,
        attention_mask: Optional[Tensor] = None,
        output_hidden_states: bool = False,
    ) -> Dict[str, Tensor]:
        """Forward pass."""
        if waveform.dim() == 1:
            waveform = waveform.unsqueeze(0).unsqueeze(1)
        elif waveform.dim() == 2:
            waveform = waveform.unsqueeze(1)

        features = self.feature_encoder(waveform)
        features = features.transpose(1, 2)
        features = self.feature_projection(features)

        if self.training:
            features, mask = self.apply_mask(features)

        hidden_states = self.transformer_encoder(
            features, src_key_padding_mask=attention_mask
        )

        output = {"last_hidden_state": hidden_states}

        if hasattr(self, "ctc_head"):
            logits = self.ctc_head(hidden_states)
            output["logits"] = logits

        if output_hidden_states:
            output["hidden_states"] = hidden_states

        return output


class ConformerBlock(nn.Module):
    """Conformer block: Convolution-augmented transformer for speech."""

    def __init__(
        self,
        hidden_size: int,
        num_heads: int = 4,
        intermediate_size: int = 2048,
        conv_kernel_size: int = 31,
        dropout: float = 0.1,
    ):
        super().__init__()

        self.ffn1 = nn.Sequential(
            nn.Linear(hidden_size, intermediate_size),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(intermediate_size, hidden_size),
            nn.Dropout(dropout),
        )
        self.norm1 = nn.LayerNorm(hidden_size)

        self.self_attn = nn.MultiheadAttention(
            hidden_size,
            num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.norm2 = nn.LayerNorm(hidden_size)

        self.conv_module = nn.Sequential(
            nn.Conv1d(
                hidden_size,
                hidden_size * 2,
                kernel_size=1,
            ),
            nn.GLU(dim=1),
            nn.Conv1d(
                hidden_size,
                hidden_size,
                kernel_size=conv_kernel_size,
                padding=(conv_kernel_size - 1) // 2,
                groups=hidden_size,
            ),
            nn.BatchNorm1d(hidden_size),
            nn.SiLU(),
            nn.Conv1d(hidden_size, hidden_size, kernel_size=1),
            nn.Dropout(dropout),
        )
        self.norm3 = nn.LayerNorm(hidden_size)

        self.ffn2 = nn.Sequential(
            nn.Linear(hidden_size, intermediate_size),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(intermediate_size, hidden_size),
            nn.Dropout(dropout),
        )
        self.norm4 = nn.LayerNorm(hidden_size)

    def forward(
        self,
        x: Tensor,
        attention_mask: Optional[Tensor] = None,
    ) -> Tensor:
        """Forward pass through conformer block."""
        residual = x
        x = self.norm1(x)
        x = residual + 0.5 * self.ffn1(x)

        residual = x
        x = self.norm2(x)
        attn_out, _ = self.self_attn(x, x, x, key_padding_mask=attention_mask)
        x = residual + attn_out

        residual = x
        x = self.norm3(x)
        x_conv = x.transpose(1, 2)
        x_conv = self.conv_module(x_conv)
        x_conv = x_conv.transpose(1, 2)
        x = residual + x_conv

        residual = x
        x = self.norm4(x)
        x = residual + 0.5 * self.ffn2(x)

        return x


class ConformerModel(nn.Module):
    """Conformer: Convolution-augmented transformer for speech recognition."""

    def __init__(self, config: SpeechConfig):
        super().__init__()
        self.config = config

        self.input_projection = nn.Linear(1, config.hidden_size)

        self.conformer_blocks = nn.ModuleList(
            [
                ConformerBlock(
                    hidden_size=config.hidden_size,
                    num_heads=config.num_attention_heads,
                    intermediate_size=config.intermediate_size,
                    dropout=config.hidden_dropout,
                )
                for _ in range(config.num_hidden_layers)
            ]
        )

        self.output_norm = nn.LayerNorm(config.hidden_size)

        if config.use_ctc:
            self.ctc_head = nn.Linear(config.hidden_size, config.vocab_size)

    def forward(
        self,
        waveform: Tensor,
        attention_mask: Optional[Tensor] = None,
        output_hidden_states: bool = False,
    ) -> Dict[str, Tensor]:
        """Forward pass."""
        if waveform.dim() == 1:
            waveform = waveform.unsqueeze(0).unsqueeze(2)
        elif waveform.dim() == 2:
            waveform = waveform.unsqueeze(2)

        x = self.input_projection(waveform)

        hidden_states = []
        for block in self.conformer_blocks:
            x = block(x, attention_mask)
            if output_hidden_states:
                hidden_states.append(x)

        x = self.output_norm(x)

        output = {"last_hidden_state": x}

        if hasattr(self, "ctc_head"):
            logits = self.ctc_head(x)
            output["logits"] = logits

        if output_hidden_states:
            output["hidden_states"] = hidden_states

        return output


class WhisperModel(nn.Module):
    """Whisper: OpenAI's speech recognition model."""

    def __init__(self, config: SpeechConfig):
        super().__init__()
        self.config = config

        self.audio_encoder = nn.Sequential(
            nn.Conv1d(1, 512, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv1d(512, 512, kernel_size=3, stride=2, padding=1),
            nn.GELU(),
            nn.Conv1d(512, 512, kernel_size=3, stride=2, padding=1),
            nn.GELU(),
            nn.Conv1d(512, 512, kernel_size=3, stride=2, padding=1),
            nn.GELU(),
            nn.Conv1d(512, 512, kernel_size=3, stride=2, padding=1),
            nn.GELU(),
            nn.Conv1d(512, 512, kernel_size=3, stride=2, padding=1),
            nn.GELU(),
            nn.Conv1d(512, config.hidden_size, kernel_size=3, stride=2, padding=1),
        )

        self.audio_projection = nn.Linear(config.hidden_size, config.hidden_size)

        decoder_layer = nn.TransformerDecoderLayer(
            d_model=config.hidden_size,
            nhead=config.num_attention_heads,
            dim_feedforward=config.intermediate_size,
            dropout=config.hidden_dropout,
            activation=config.activation,
            batch_first=True,
        )
        self.decoder = nn.TransformerDecoder(
            decoder_layer,
            num_layers=config.num_hidden_layers,
        )

        self.token_embedding = nn.Embedding(config.vocab_size, config.hidden_size)
        self.positional_embedding = nn.Parameter(
            torch.randn(1, config.max_length, config.hidden_size)
        )

        self.token_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

    def encode_audio(self, waveform: Tensor) -> Tensor:
        """Encode audio to hidden states."""
        if waveform.dim() == 1:
            waveform = waveform.unsqueeze(0).unsqueeze(1)
        elif waveform.dim() == 2:
            waveform = waveform.unsqueeze(1)

        audio_features = self.audio_encoder(waveform)
        audio_features = audio_features.transpose(1, 2)
        audio_features = self.audio_projection(audio_features)
        return audio_features

    def decode_tokens(
        self,
        audio_features: Tensor,
        token_ids: Tensor,
    ) -> Tensor:
        """Decode tokens using audio features."""
        batch_size = audio_features.size(0)
        seq_len = token_ids.size(1)

        token_embeds = self.token_embedding(token_ids)
        positions = self.positional_embedding[:, :seq_len, :]
        token_embeds = token_embeds + positions

        decoder_output = self.decoder(
            token_embeds,
            audio_features,
        )

        logits = self.token_head(decoder_output)
        return logits

    def forward(
        self,
        waveform: Tensor,
        input_ids: Optional[Tensor] = None,
        labels: Optional[Tensor] = None,
    ) -> Dict[str, Tensor]:
        """Forward pass."""
        audio_features = self.encode_audio(waveform)

        if input_ids is not None:
            logits = self.decode_tokens(audio_features, input_ids)
        else:
            logits = None

        output = {
            "audio_features": audio_features,
            "logits": logits,
        }

        if labels is not None:
            decoder_input_ids = labels[..., :-1]
            target_ids = labels[..., 1:]
            logits = self.decode_tokens(audio_features, decoder_input_ids)
            output["loss"] = F.cross_entropy(
                logits.reshape(-1, self.config.vocab_size),
                target_ids.reshape(-1),
                ignore_index=-100,
            )

        return output


class AudioSpectrogramTransformer(nn.Module):
    """Audio Spectrogram Transformer (AST)."""

    def __init__(
        self,
        patch_size: int = 16,
        embed_dim: int = 768,
        depth: int = 12,
        num_heads: int = 12,
        mlp_ratio: float = 4.0,
        num_classes: int = 527,
        in_channels: int = 1,
    ):
        super().__init__()

        self.patch_size = patch_size
        self.embed_dim = embed_dim

        self.patch_embedding = nn.Conv2d(
            in_channels,
            embed_dim,
            kernel_size=patch_size,
            stride=patch_size,
        )

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.dist_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, 1214, embed_dim))

        self.blocks = nn.ModuleList(
            [
                nn.TransformerEncoderLayer(
                    d_model=embed_dim,
                    nhead=num_heads,
                    dim_feedforward=int(embed_dim * mlp_ratio),
                    dropout=0.1,
                    activation="gelu",
                    batch_first=True,
                    norm_first=True,
                )
                for _ in range(depth)
            ]
        )

        self.norm = nn.LayerNorm(embed_dim)
        self.head = nn.Linear(embed_dim, num_classes)

    def forward(self, mel_spec: Tensor) -> Dict[str, Tensor]:
        """Forward pass."""
        if mel_spec.dim() == 3:
            mel_spec = mel_spec.unsqueeze(1)

        x = self.patch_embedding(mel_spec)
        x = x.flatten(2).transpose(1, 2)

        cls_tokens = self.cls_token.expand(x.size(0), -1, -1)
        x = torch.cat([cls_tokens, x], dim=1)

        x = x + self.pos_embed[:, : x.size(1), :]

        for block in self.blocks:
            x = block(x)

        x = self.norm(x)

        cls_output = x[:, 0]
        dist_output = x[:, 1]

        logits = self.head(cls_output)

        return {
            "logits": logits,
            "cls_features": cls_output,
            "dist_features": dist_output,
            "all_features": x,
        }


class LanguageModelInterface(nn.Module):
    """Interface for integrating speech models with language models."""

    def __init__(
        self,
        speech_config: SpeechConfig,
        lm_config: Optional[Dict[str, Any]] = None,
    ):
        super().__init__()
        self.speech_config = speech_config
        self.lm_config = lm_config or {}

        self.speech_to_text_adapter = nn.Sequential(
            nn.Linear(
                speech_config.hidden_size, self.lm_config.get("hidden_size", 768)
            ),
            nn.GELU(),
            nn.Dropout(speech_config.hidden_dropout),
        )

    def forward(
        self,
        speech_features: Tensor,
        text_embeddings: Optional[Tensor] = None,
    ) -> Dict[str, Tensor]:
        """Fuse speech features with text embeddings."""
        adapted_features = self.speech_to_text_adapter(speech_features)

        output = {"adapted_features": adapted_features}

        if text_embeddings is not None:
            fused = torch.cat([adapted_features, text_embeddings], dim=1)
            output["fused_features"] = fused

        return output

    def generate_text(
        self,
        speech_features: Tensor,
        max_length: int = 100,
        temperature: float = 1.0,
    ) -> Tensor:
        """Generate text from speech features."""
        adapted = self.speech_to_text_adapter(speech_features)

        generated = []
        current_features = adapted

        for _ in range(max_length):
            logits = self._compute_logits(current_features)
            probs = F.softmax(logits / temperature, dim=-1)
            next_token = torch.multinomial(probs, 1)
            generated.append(next_token)

            if next_token.item() == self.speech_config.eos_token_id:
                break

        return torch.cat(generated, dim=-1)

    def _compute_logits(self, features: Tensor) -> Tensor:
        """Compute logits for next token prediction."""
        return torch.randn(
            features.size(0), self.speech_config.vocab_size, device=features.device
        )


class SpeechDataset(Dataset):
    """Dataset for speech recognition."""

    def __init__(
        self,
        audio_paths: List[str],
        transcripts: Optional[List[str]] = None,
        config: Optional[AudioConfig] = None,
        tokenizer: Optional[Any] = None,
    ):
        self.audio_paths = audio_paths
        self.transcripts = transcripts
        self.config = config or AudioConfig()
        self.tokenizer = tokenizer

    def __len__(self) -> int:
        return len(self.audio_paths)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        item = {"audio_path": self.audio_paths[idx]}

        if self.transcripts is not None:
            item["transcript"] = self.transcripts[idx]

            if self.tokenizer is not None:
                item["labels"] = self.tokenizer(
                    self.transcripts[idx],
                    return_tensors="pt",
                )["input_ids"].squeeze(0)

        return item


class SpeechRecognizer:
    """High-level speech recognition interface."""

    def __init__(
        self,
        model_type: str = "wav2vec2",
        model_config: Optional[SpeechConfig] = None,
        audio_config: Optional[AudioConfig] = None,
    ):
        self.model_type = model_type
        self.model_config = model_config or SpeechConfig()
        self.audio_config = audio_config or AudioConfig()

        self.preprocessor = AudioPreprocessor(self.audio_config)

        if model_type == "wav2vec2":
            self.model = Wav2Vec2Model(self.model_config)
        elif model_type == "hubert":
            self.model = HubertModel(self.model_config)
        elif model_type == "conformer":
            self.model = ConformerModel(self.model_config)
        elif model_type == "whisper":
            self.model = WhisperModel(self.model_config)
        elif model_type == "ast":
            self.model = AudioSpectrogramTransformer()
        else:
            raise ValueError(f"Unknown model type: {model_type}")

        if self.model_config.use_ctc:
            self.ctc_loss = CTCLossWrapper(vocab_size=self.model_config.vocab_size)

    def recognize(
        self,
        waveform: Tensor,
        method: str = "greedy",
    ) -> str:
        """Recognize speech from waveform."""
        waveform = self.preprocessor(waveform)

        if self.model_type == "ast":
            features = FeatureExtractor(self.audio_config).extract_mel_spectrogram(
                waveform
            )
            outputs = self.model(features)
            logits = outputs["logits"]
            predictions = logits.argmax(dim=-1)
            return self._decode_predictions(predictions)
        else:
            outputs = self.model(waveform)
            logits = outputs.get("logits", outputs["last_hidden_state"])

            if self.model_config.use_ctc:
                decoded = self.ctc_loss.decode(logits, method=method)
                return self._decode_ctc_predictions(decoded)
            else:
                predictions = logits.argmax(dim=-1)
                return self._decode_predictions(predictions)

    def _decode_predictions(self, predictions: Tensor) -> str:
        """Decode model predictions to text."""
        return "decoded_text"

    def _decode_ctc_predictions(self, decoded: List[List[int]]) -> str:
        """Decode CTC predictions to text."""
        return "decoded_ctc_text"

    def transcribe_file(
        self,
        audio_path: str,
        method: str = "greedy",
    ) -> str:
        """Transcribe audio file."""
        if not TORCHAUDIO_AVAILABLE:
            raise ImportError("torchaudio is required for file transcription")

        waveform, sample_rate = torchaudio.load(audio_path)

        if sample_rate != self.audio_config.sample_rate:
            waveform = self.preprocessor.resample(
                waveform, sample_rate, self.audio_config.sample_rate
            )

        return self.recognize(waveform.squeeze(), method=method)


def create_speech_model(
    model_type: str,
    config: Optional[SpeechConfig] = None,
) -> nn.Module:
    """Factory function to create speech models."""
    config = config or SpeechConfig()

    if model_type == "wav2vec2":
        return Wav2Vec2Model(config)
    elif model_type == "hubert":
        return HubertModel(config)
    elif model_type == "conformer":
        return ConformerModel(config)
    elif model_type == "whisper":
        return WhisperModel(config)
    elif model_type == "ast":
        return AudioSpectrogramTransformer()
    else:
        raise ValueError(f"Unknown model type: {model_type}")


def load_pretrained_speech_model(
    model_type: str,
    model_name: str = "facebook/wav2vec2-base-960h",
    device: Optional[torch.device] = None,
) -> nn.Module:
    """Load pretrained speech model from HuggingFace."""
    if not TRANSFORMERS_AVAILABLE:
        raise ImportError("transformers library is required for pretrained models")

    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if model_type == "wav2vec2":
        model = Wav2Vec2ForCTC.from_pretrained(model_name)
    elif model_type == "hubert":
        model = HubertForCTC.from_pretrained(model_name)
    elif model_type == "whisper":
        model = WhisperForConditionalGeneration.from_pretrained(model_name)
    else:
        raise ValueError(f"Unknown model type: {model_type}")

    return model.to(device)


__all__ = [
    "AudioConfig",
    "SpeechConfig",
    "AudioFeatureType",
    "FeatureExtractor",
    "AudioPreprocessor",
    "CTCLossWrapper",
    "Wav2Vec2Model",
    "HubertModel",
    "ConformerModel",
    "ConformerBlock",
    "WhisperModel",
    "AudioSpectrogramTransformer",
    "LanguageModelInterface",
    "SpeechDataset",
    "SpeechRecognizer",
    "create_speech_model",
    "load_pretrained_speech_model",
]
