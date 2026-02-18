"""
Comprehensive Speech Recognition Module for Fishstick

This module provides a complete speech recognition pipeline including:
- Acoustic models (DeepSpeech, LAS, CTC, RNN-T, Conformer, Wav2Vec2, HuBERT, Whisper)
- Feature extractors (LogMel, MFCC, FilterBank, RawAudio, Wav2Vec2 features)
- Decoding strategies (Greedy, Beam Search, CTC Beam, KenLM, Neural LM)
- Language models (N-gram, Neural, Transformer, RNN)
- Text processing (Normalization, BPE, WordPiece, SentencePiece)
- Training utilities and evaluation metrics

Author: Fishstick AI Team
"""

import math
import warnings
from abc import ABC, abstractmethod
from collections import defaultdict
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset

warnings.filterwarnings("ignore")


# =============================================================================
# CONFIGURATION AND UTILITIES
# =============================================================================


@dataclass
class RecognitionConfig:
    """Configuration for speech recognition models."""

    sample_rate: int = 16000
    n_fft: int = 400
    hop_length: int = 160
    win_length: int = 400
    n_mels: int = 80
    n_mfcc: int = 13
    vocab_size: int = 29
    hidden_dim: int = 512
    num_layers: int = 5
    dropout: float = 0.3
    batch_size: int = 32
    learning_rate: float = 1e-3
    max_grad_norm: float = 5.0
    beam_width: int = 10
    lm_weight: float = 0.5
    ctc_weight: float = 0.3
    blank_threshold: float = 0.5


def _get_activation(activation: str = "relu") -> nn.Module:
    activations = {
        "relu": nn.ReLU(),
        "gelu": nn.GELU(),
        "swish": nn.SiLU(),
        "tanh": nn.Tanh(),
        "sigmoid": nn.Sigmoid(),
    }
    return activations.get(activation, nn.ReLU())


def _compute_padding(kernel_size: int, dilation: int = 1) -> int:
    return (kernel_size - 1) * dilation // 2


# =============================================================================
# FEATURE EXTRACTORS
# =============================================================================


class FeatureExtractor(ABC, nn.Module):
    """Abstract base class for audio feature extractors."""

    @abstractmethod
    def forward(self, x: Tensor) -> Tensor:
        """Extract features from audio."""
        pass

    @property
    @abstractmethod
    def output_dim(self) -> int:
        pass


class LogMelSpectrogram(FeatureExtractor):
    """Log Mel Spectrogram feature extractor."""

    def __init__(
        self,
        sample_rate: int = 16000,
        n_fft: int = 400,
        hop_length: int = 160,
        win_length: int = 400,
        n_mels: int = 80,
        f_min: float = 0.0,
        f_max: Optional[float] = None,
    ):
        super().__init__()
        self.sample_rate = sample_rate
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.win_length = win_length
        self.n_mels = n_mels
        self.f_min = f_min
        self.f_max = f_max or sample_rate // 2

        mel_fb = self._create_mel_filterbank()
        self.register_buffer("mel_fb", mel_fb)
        window = torch.hann_window(win_length)
        self.register_buffer("window", window)

    def _create_mel_filterbank(self) -> Tensor:
        def hz_to_mel(hz: float) -> float:
            return 2595 * math.log10(1 + hz / 700)

        def mel_to_hz(mel: float) -> float:
            return 700 * (10 ** (mel / 2595) - 1)

        fft_freqs = torch.linspace(0, self.sample_rate / 2, self.n_fft // 2 + 1)
        mel_min = hz_to_mel(self.f_min)
        mel_max = hz_to_mel(self.f_max)
        mel_bins = torch.linspace(mel_min, mel_max, self.n_mels + 2)
        freq_bins = torch.tensor([mel_to_hz(m) for m in mel_bins])

        mel_fb = torch.zeros(self.n_mels, self.n_fft // 2 + 1)
        for i in range(self.n_mels):
            left = freq_bins[i]
            center = freq_bins[i + 1]
            right = freq_bins[i + 2]
            for j, f in enumerate(fft_freqs):
                if left <= f <= center:
                    mel_fb[i, j] = (f - left) / (center - left)
                elif center < f <= right:
                    mel_fb[i, j] = (right - f) / (right - center)
        return mel_fb

    def forward(self, x: Tensor) -> Tensor:
        if x.dim() == 3:
            x = x.squeeze(1)

        if x.shape[-1] < self.win_length:
            x = F.pad(x, (0, self.win_length - x.shape[-1]))

        stft = torch.stft(
            x,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            win_length=self.win_length,
            window=self.window,
            center=True,
            pad_mode="reflect",
            normalized=False,
            onesided=True,
            return_complex=True,
        )

        spec = torch.abs(stft)
        mel_spec = torch.matmul(self.mel_fb, spec)
        mel_spec = torch.log(mel_spec + 1e-10)
        mel_spec = mel_spec.transpose(1, 2)
        return mel_spec

    @property
    def output_dim(self) -> int:
        return self.n_mels


class MFCC(FeatureExtractor):
    """MFCC feature extractor."""

    def __init__(
        self,
        sample_rate: int = 16000,
        n_fft: int = 400,
        hop_length: int = 160,
        win_length: int = 400,
        n_mels: int = 80,
        n_mfcc: int = 13,
        f_min: float = 0.0,
        f_max: Optional[float] = None,
    ):
        super().__init__()
        self.log_mel = LogMelSpectrogram(
            sample_rate, n_fft, hop_length, win_length, n_mels, f_min, f_max
        )
        self.n_mfcc = n_mfcc
        dct_matrix = self._create_dct_matrix(n_mels, n_mfcc)
        self.register_buffer("dct_matrix", dct_matrix)

    def _create_dct_matrix(self, n_input: int, n_output: int) -> Tensor:
        dct = torch.zeros(n_output, n_input)
        for i in range(n_output):
            for j in range(n_input):
                if i == 0:
                    dct[i, j] = 1 / math.sqrt(n_input)
                else:
                    dct[i, j] = math.sqrt(2 / n_input) * math.cos(
                        (math.pi / n_input) * (j + 0.5) * i
                    )
        return dct

    def forward(self, x: Tensor) -> Tensor:
        log_mel = self.log_mel(x)
        mfcc = torch.matmul(log_mel, self.dct_matrix.T)
        return mfcc

    @property
    def output_dim(self) -> int:
        return self.n_mfcc


class FilterBank(FeatureExtractor):
    """Filter bank feature extractor."""

    def __init__(
        self,
        sample_rate: int = 16000,
        n_fft: int = 400,
        hop_length: int = 160,
        win_length: int = 400,
        n_mels: int = 80,
        f_min: float = 0.0,
        f_max: Optional[float] = None,
    ):
        super().__init__()
        self.log_mel = LogMelSpectrogram(
            sample_rate, n_fft, hop_length, win_length, n_mels, f_min, f_max
        )

    def forward(self, x: Tensor) -> Tensor:
        if x.dim() == 3:
            x = x.squeeze(1)

        log_mel = self.log_mel.log_mel_fb  # Get the mel filterbank

        if x.shape[-1] < self.log_mel.win_length:
            x = F.pad(x, (0, self.log_mel.win_length - x.shape[-1]))

        stft = torch.stft(
            x,
            n_fft=self.log_mel.n_fft,
            hop_length=self.log_mel.hop_length,
            win_length=self.log_mel.win_length,
            window=self.log_mel.window,
            center=True,
            pad_mode="reflect",
            normalized=False,
            onesided=True,
            return_complex=True,
        )

        spec = torch.abs(stft)
        fbank = torch.matmul(self.log_mel.mel_fb, spec)
        fbank = fbank.transpose(1, 2)
        return fbank

    @property
    def output_dim(self) -> int:
        return self.log_mel.n_mels


class RawAudio(FeatureExtractor):
    """Raw audio feature extractor using CNN layers."""

    def __init__(
        self,
        output_dim: int = 512,
        kernel_sizes: List[int] = [10, 3, 3, 3, 3, 2, 2],
        strides: List[int] = [5, 2, 2, 2, 2, 2, 2],
    ):
        super().__init__()
        self.output_dim = output_dim
        in_channels = 1
        channels = [output_dim // 4, output_dim // 2, output_dim]

        layers = []
        for i, (k, s) in enumerate(zip(kernel_sizes, strides)):
            out_ch = channels[min(i, len(channels) - 1)]
            layers.extend(
                [
                    nn.Conv1d(in_channels, out_ch, k, s, _compute_padding(k)),
                    nn.LayerNorm(out_ch),
                    nn.GELU(),
                    nn.Dropout(0.1),
                ]
            )
            in_channels = out_ch

        self.conv_layers = nn.Sequential(*layers)
        self.proj = nn.Linear(in_channels, output_dim)

    def forward(self, x: Tensor) -> Tensor:
        if x.dim() == 2:
            x = x.unsqueeze(1)
        x = self.conv_layers(x)
        x = x.transpose(1, 2)
        x = self.proj(x)
        return x

    @property
    def output_dim(self) -> int:
        return self.output_dim


class Wav2Vec2Feature(FeatureExtractor):
    """Wav2Vec2-style feature extractor."""

    def __init__(
        self,
        output_dim: int = 512,
        num_layers: int = 7,
        kernel_sizes: List[int] = [10, 3, 3, 3, 3, 2, 2],
        strides: List[int] = [5, 2, 2, 2, 2, 2, 2],
    ):
        super().__init__()
        self.output_dim = output_dim
        in_channels = 1
        conv_layers = []

        for i, (k, s) in enumerate(
            zip(kernel_sizes[:num_layers], strides[:num_layers])
        ):
            out_ch = min(512, 64 * (2 ** (i // 2)))
            conv_layers.extend(
                [
                    nn.Conv1d(in_channels, out_ch, k, s, bias=False),
                    nn.Dropout(0.0),
                ]
            )
            if i < num_layers - 1:
                conv_layers.append(nn.GELU())
            in_channels = out_ch

        self.feature_encoder = nn.Sequential(*conv_layers)
        self.layer_norm = nn.LayerNorm(in_channels)
        self.projection = nn.Linear(in_channels, output_dim)

    def forward(self, x: Tensor) -> Tensor:
        if x.dim() == 2:
            x = x.unsqueeze(1)
        features = self.feature_encoder(x)
        features = features.transpose(1, 2)
        features = self.layer_norm(features)
        features = self.projection(features)
        return features

    @property
    def output_dim(self) -> int:
        return self.output_dim


# =============================================================================
# ACOUSTIC MODELS
# =============================================================================

class AcousticModel(ABC, nn.Module):
    """Abstract base class for acoustic models."""
    
    @abstractmethod
    def forward(
        self,
        features: Tensor,
        feature_lengths: Optional[Tensor] = None
    ) -> Union[Tensor, Tuple[Tensor, Tensor]]:
        """Forward pass of acoustic model."""
        pass


class DeepSpeech(AcousticModel):
    """DeepSpeech: CTC-based speech recognition model."""
    
    def __init__(
        self,
        input_dim: int,
        vocab_size: int,
        hidden_dim: int = 2048,
        num_layers: int = 5,
        dropout: float = 0.3,
        bidirectional: bool = True,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.vocab_size = vocab_size
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        
        self.layer_norm = nn.LayerNorm(input_dim)
        self.rnns = nn.ModuleList()
        rnn_input_dim = input_dim
        
        for i in range(num_layers):
            self.rnns.append(
                nn.LSTM(
                    rnn_input_dim,
                    hidden_dim,
                    num_layers=1,
                    batch_first=True,
                    bidirectional=bidirectional,
                )
            )
            rnn_output_dim = hidden_dim * 2 if bidirectional else hidden_dim
            self.rnns.append(nn.BatchNorm1d(rnn_output_dim))
            rnn_input_dim = rnn_output_dim
        
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(rnn_output_dim, vocab_size)
    
    def forward(
        self,
        features: Tensor,
        feature_lengths: Optional[Tensor] = None
    ) -> Tuple[Tensor, Tensor]:
        """Forward pass."""
        x = self.layer_norm(features)
        
        for i in range(0, len(self.rnns), 2):
            rnn = self.rnns[i]
            bn = self.rnns[i + 1]
            
            x, _ = rnn(x)
            x = x.transpose(1, 2)
            x = bn(x)
            x = x.transpose(1, 2)
            x = self.dropout(x)
        
        logits = self.classifier(x)
        
        if feature_lengths is not None:
            output_lengths = feature_lengths
        else:
            output_lengths = torch.full((x.size(0),), x.size(1), dtype=torch.long)
        
        return logits, output_lengths


class ListenAttendSpell(AcousticModel):
    """Listen, Attend and Spell (LAS) model."""
    
    def __init__(
        self,
        input_dim: int,
        vocab_size: int,
        encoder_dim: int = 512,
        decoder_dim: int = 512,
        attention_dim: int = 128,
        num_encoder_layers: int = 3,
        num_decoder_layers: int = 1,
        dropout: float = 0.3,
    ):
        super().__init__()
        self.vocab_size = vocab_size
        
        # Encoder (Listener)
        self.encoder = nn.LSTM(
            input_dim,
            encoder_dim,
            num_layers=num_encoder_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_encoder_layers > 1 else 0,
        )
        
        # Decoder (Speller)
        self.decoder = nn.LSTM(
            decoder_dim,
            decoder_dim,
            num_layers=num_decoder_layers,
            batch_first=True,
            dropout=dropout if num_decoder_layers > 1 else 0,
        )
        
        # Attention
        self.attention = nn.MultiheadAttention(
            encoder_dim * 2,
            num_heads=8,
            dropout=dropout,
            batch_first=True,
        )
        
        # Embeddings
        self.embedding = nn.Embedding(vocab_size, decoder_dim)
        
        # Output projection
        self.output_proj = nn.Linear(decoder_dim + encoder_dim * 2, vocab_size)
        
    def forward(
        self,
        features: Tensor,
        feature_lengths: Optional[Tensor] = None,
        targets: Optional[Tensor] = None,
    ) -> Union[Tensor, Tuple[Tensor, Tensor]]:
        """Forward pass."""
        # Encode
        encoder_out, _ = self.encoder(features)
        
        if targets is None:
            # Inference mode - greedy decoding
            return self._greedy_decode(encoder_out, feature_lengths)
        
        # Training mode
        batch_size = features.size(0)
        max_len = targets.size(1)
        
        # Teacher forcing
        embedded = self.embedding(targets[:, :-1])
        decoder_out, _ = self.decoder(embedded)
        
        # Attention
        attended, _ = self.attention(decoder_out, encoder_out, encoder_out)
        
        # Combine and predict
        combined = torch.cat([decoder_out, attended], dim=-1)
        logits = self.output_proj(combined)
        
        return logits
    
    def _greedy_decode(
        self,
        encoder_out: Tensor,
        lengths: Optional[Tensor],
        max_len: int = 100,
    ) -> Tensor:
        """Greedy decoding for inference."""
        batch_size = encoder_out.size(0)
        device = encoder_out.device
        
        # Start with BOS token (assume 0)
        decoder_input = torch.zeros(batch_size, 1, dtype=torch.long, device=device)
        outputs = []
        
        hidden = None
        for _ in range(max_len):
            embedded = self.embedding(decoder_input[:, -1:])
            decoder_out, hidden = self.decoder(embedded, hidden)
            
            attended, _ = self.attention(decoder_out, encoder_out, encoder_out)
            combined = torch.cat([decoder_out, attended], dim=-1)
            logits = self.output_proj(combined)
            
            _, predicted = logits.max(dim=-1)
            outputs.append(predicted)
            
            decoder_input = torch.cat([decoder_input, predicted], dim=1)
        
        return torch.cat(outputs, dim=1)


class CTCModel(AcousticModel):
    """Pure CTC acoustic model with convolutions."""
    
    def __init__(
        self,
        input_dim: int,
        vocab_size: int,
        hidden_dim: int = 512,
        num_conv_layers: int = 3,
        num_rnn_layers: int = 5,
        dropout: float = 0.3,
    ):
        super().__init__()
        
        # Convolutional frontend
        conv_layers = []
        in_channels = 1
        for i in range(num_conv_layers):
            out_channels = min(256, 32 * (2 ** i))
            conv_layers.extend([
                nn.Conv2d(in_channels, out_channels, 3, stride=2, padding=1),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(),
                nn.Dropout2d(dropout / 2),
            ])
            in_channels = out_channels
        
        self.conv_layers = nn.Sequential(*conv_layers)
        
        # Calculate conv output dim
        self.conv_out_channels = in_channels
        self.conv_out_dim = input_dim // (2 ** num_conv_layers)
        
        # RNN layers
        self.rnns = nn.ModuleList()
        rnn_input_dim = self.conv_out_channels * self.conv_out_dim
        
        for i in range(num_rnn_layers):
            self.rnns.append(
                nn.LSTM(
                    rnn_input_dim,
                    hidden_dim,
                    num_layers=1,
                    batch_first=True,
                    bidirectional=True,
                )
            )
            rnn_input_dim = hidden_dim * 2
        
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(hidden_dim * 2, vocab_size)
    
    def forward(
        self,
        features: Tensor,
        feature_lengths: Optional[Tensor] = None
    ) -> Tuple[Tensor, Tensor]:
        """Forward pass."""
        batch_size, time, freq = features.shape
        
        # Add channel dim and apply convolutions
        x = features.unsqueeze(1)
        x = self.conv_layers(x)
        
        # Reshape for RNN
        b, c, t, f = x.shape
        x = x.permute(0, 2, 1, 3).contiguous()
        x = x.view(b, t, c * f)
        
        # Apply RNNs
        for rnn in self.rnns:
            x, _ = rnn(x)
            x = self.dropout(x)
        
        logits = self.classifier(x)
        
        # Calculate output lengths
        if feature_lengths is not None:
            output_lengths = feature_lengths // (2 ** len(self.conv_layers) // 3)
            output_lengths = torch.clamp(output_lengths, min=1, max=logits.size(1))
        else:
            output_lengths = torch.full((batch_size,), logits.size(1), dtype=torch.long)
        
        return logits, output_lengths


class RNNTransducer(AcousticModel):
    """RNN Transducer (RNN-T) model."""
    
    def __init__(
        self,
        input_dim: int,
        vocab_size: int,
        encoder_dim: int = 512,
        predictor_dim: int = 512,
        joint_dim: int = 512,
        num_encoder_layers: int = 5,
        num_predictor_layers: int = 2,
        dropout: float = 0.3,
    ):
        super().__init__()
        self.vocab_size = vocab_size
        
        # Encoder (audio encoder)
        self.encoder = nn.LSTM(
            input_dim,
            encoder_dim,
            num_layers=num_encoder_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_encoder_layers > 1 else 0,
        )
        
        # Predictor (label encoder)
        self.embedding = nn.Embedding(vocab_size, predictor_dim)
        self.predictor = nn.LSTM(
            predictor_dim,
            predictor_dim,
            num_layers=num_predictor_layers,
            batch_first=True,
            dropout=dropout if num_predictor_layers > 1 else 0,
        )
        
        # Joint network
        self.joint_fc1 = nn.Linear(encoder_dim * 2 + predictor_dim, joint_dim)
        self.joint_fc2 = nn.Linear(joint_dim, vocab_size)
        self.dropout = nn.Dropout(dropout)
    
    def forward(
        self,
        features: Tensor,
        feature_lengths: Optional[Tensor] = None,
        targets: Optional[Tensor] = None,
    ) -> Union[Tensor, Tuple[Tensor, Tensor, Tensor]]:
        """Forward pass."""
        # Encode audio
        encoder_out, _ = self.encoder(features)
        
        if targets is None:
            # Inference mode
            return self._greedy_decode(encoder_out, feature_lengths)
        
        # Encode labels
        embedded = self.embedding(targets)
        predictor_out, _ = self.predictor(embedded)
        
        # Joint network
        # Expand dimensions for broadcasting
        encoder_out_exp = encoder_out.unsqueeze(2)  # (B, T, 1, E)
        predictor_out_exp = predictor_out.unsqueeze(1)  # (B, 1, U, P)
        
        # Combine
        joint_input = torch.cat([
            encoder_out_exp.expand(-1, -1, predictor_out.size(1), -1),
            predictor_out_exp.expand(-1, encoder_out.size(1), -1, -1),
        ], dim=-1)
        
        joint_hidden = torch.tanh(self.joint_fc1(joint_input))
        joint_hidden = self.dropout(joint_hidden)
        logits = self.joint_fc2(joint_hidden)
        
        return logits, encoder_out, predictor_out
    
    def _greedy_decode(
        self,
        encoder_out: Tensor,
        lengths: Optional[Tensor],
        max_len: int = 100,
    ) -> Tensor:
        """Greedy decoding."""
        batch_size = encoder_out.size(0)
        device = encoder_out.device
        
        predictions = []
        predictor_state = None
        
        for t in range(encoder_out.size(1)):
            # Get current encoder output
            enc_t = encoder_out[:, t:t+1, :]
            
            # Start with blank
            label = torch.zeros(batch_size, 1, dtype=torch.long, device=device)
            
            for _ in range(max_len):
                emb = self.embedding(label[:, -1:])
                pred_out, predictor_state = self.predictor(emb, predictor_state)
                
                # Joint
                joint_input = torch.cat([enc_t, pred_out], dim=-1)
                joint_hidden = torch.tanh(self.joint_fc1(joint_input))
                logits = self.joint_fc2(joint_hidden)
                
                _, label_next = logits.max(dim=-1)
                
                if label_next.squeeze() == 0:  # Blank token
                    break
                
                predictions.append(label_next)
                label = torch.cat([label, label_next], dim=1)
        
        if predictions:
            return torch.cat(predictions, dim=1)
        return torch.zeros(batch_size, 1, dtype=torch.long, device=device)


class Conformer(AcousticModel):
    """Conformer: Convolution-augmented Transformer for ASR."""
    
    def __init__(
        self,
        input_dim: int,
        vocab_size: int,
        d_model: int = 256,
        num_heads: int = 8,
        num_layers: int = 12,
        conv_kernel_size: int = 31,
        dropout: float = 0.1,
        feed_forward_expansion_factor: int = 4,
    ):
        super().__init__()
        self.d_model = d_model
        
        # Input projection
        self.input_proj = nn.Linear(input_dim, d_model)
        
        # Conformer blocks
        self.layers = nn.ModuleList([
            ConformerBlock(
                d_model=d_model,
                num_heads=num_heads,
                conv_kernel_size=conv_kernel_size,
                dropout=dropout,
                feed_forward_expansion_factor=feed_forward_expansion_factor,
            )
            for _ in range(num_layers)
        ])
        
        self.classifier = nn.Linear(d_model, vocab_size)
        self.dropout = nn.Dropout(dropout)
    
    def forward(
        self,
        features: Tensor,
        feature_lengths: Optional[Tensor] = None
    ) -> Tuple[Tensor, Tensor]:
        """Forward pass."""
        x = self.input_proj(features)
        x = self.dropout(x)
        
        # Apply conformer layers
        for layer in self.layers:
            x = layer(x)
        
        logits = self.classifier(x)
        
        if feature_lengths is not None:
            output_lengths = feature_lengths
        else:
            output_lengths = torch.full((x.size(0),), x.size(1), dtype=torch.long)
        
        return logits, output_lengths


class ConformerBlock(nn.Module):
    """Single Conformer block."""
    
    def __init__(
        self,
        d_model: int,
        num_heads: int,
        conv_kernel_size: int,
        dropout: float,
        feed_forward_expansion_factor: int,
    ):
        super().__init__()
        
        # Feed-forward module (first half)
        self.ffn1 = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model * feed_forward_expansion_factor),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * feed_forward_expansion_factor, d_model),
            nn.Dropout(dropout),
        )
        
        # Multi-head self-attention
        self.attn_norm = nn.LayerNorm(d_model)
        self.attn = nn.MultiheadAttention(
            d_model, num_heads, dropout=dropout, batch_first=True
        )
        self.attn_dropout = nn.Dropout(dropout)
        
        # Convolution module
        self.conv = ConvolutionModule(d_model, conv_kernel_size, dropout)
        
        # Feed-forward module (second half)
        self.ffn2 = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model * feed_forward_expansion_factor),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * feed_forward_expansion_factor, d_model),
            nn.Dropout(dropout),
        )
        
        self.final_norm = nn.LayerNorm(d_model)
    
    def forward(self, x: Tensor) -> Tensor:
        """Forward with residual connections."""
        # FFN1 (0.5 factor from paper)
        x = x + 0.5 * self.ffn1(x)
        
        # Self-attention
        residual = x
        x = self.attn_norm(x)
        x, _ = self.attn(x, x, x)
        x = self.attn_dropout(x)
        x = x + residual
        
        # Convolution
        x = x + self.conv(x)
        
        # FFN2
        x = x + 0.5 * self.ffn2(x)
        
        return self.final_norm(x)


class ConvolutionModule(nn.Module):
    """Conformer convolution module."""
    
    def __init__(self, d_model: int, kernel_size: int, dropout: float):
        super().__init__()
        self.layer_norm = nn.LayerNorm(d_model)
        
        self.pointwise_conv1 = nn.Conv1d(d_model, 2 * d_model, 1)
        self.depthwise_conv = nn.Conv1d(
            d_model,
            d_model,
            kernel_size,
            padding=(kernel_size - 1) // 2,
            groups=d_model,
        )
        self.batch_norm = nn.BatchNorm1d(d_model)
        self.pointwise_conv2 = nn.Conv1d(d_model, d_model, 1)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x: Tensor) -> Tensor:
        """Forward pass."""
        x = self.layer_norm(x)
        x = x.transpose(1, 2)
        
        # Pointwise expansion
        x = self.pointwise_conv1(x)
        x = F.glu(x, dim=1)
        
        # Depthwise convolution
        x = self.depthwise_conv(x)
        x = self.batch_norm(x)
        x = F.silu(x)
        
        # Pointwise projection
        x = self.pointwise_conv2(x)
        x = self.dropout(x)
        
        return x.transpose(1, 2)


class Wav2Vec2ASR(AcousticModel):
    """Wav2Vec2 for ASR with fine-tuning."""
    
    def __init__(
        self,
        vocab_size: int,
        d_model: int = 768,
        num_layers: int = 12,
        num_heads: int = 12,
        dropout: float = 0.1,
        conv_dim: int = 512,
    ):
        super().__init__()
        self.vocab_size = vocab_size
        
        # Feature encoder (CNN)
        self.feature_extractor = Wav2Vec2Feature(output_dim=conv_dim)
        
        # Feature projection
        self.feature_projection = nn.Sequential(
            nn.LayerNorm(conv_dim),
            nn.Linear(conv_dim, d_model),
            nn.Dropout(dropout),
        )
        
        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=num_heads,
            dim_feedforward=d_model * 4,
            dropout=dropout,
            activation='gelu',
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers)
        
        # Output layer
        self.classifier = nn.Linear(d_model, vocab_size)
    
    def forward(
        self,
        features: Tensor,
        feature_lengths: Optional[Tensor] = None
    ) -> Tuple[Tensor, Tensor]:
        """Forward pass."""
        # Extract features from raw audio if needed
        if features.dim() == 2 or (features.dim() == 3 and features.size(1) == 1):
            x = self.feature_extractor(features)
        else:
            x = features
        
        # Project
        x = self.feature_projection(x)
        
        # Apply transformer
        x = self.transformer(x)
        
        # Classify
        logits = self.classifier(x)
        
        if feature_lengths is not None:
            output_lengths = feature_lengths
        else:
            output_lengths = torch.full((x.size(0),), x.size(1), dtype=torch.long)
        
        return logits, output_lengths


class HuBERT(AcousticModel):
    """HuBERT: Hidden Unit BERT for ASR."""
    
    def __init__(
        self,
        vocab_size: int,
        d_model: int = 768,
        num_layers: int = 12,
        num_heads: int = 12,
        dropout: float = 0.1,
        conv_dim: int = 512,
    ):
        super().__init__()
        self.vocab_size = vocab_size
        
        # Feature encoder
        self.feature_extractor = Wav2Vec2Feature(output_dim=conv_dim)
        
        # Projection
        self.post_extract_proj = nn.Linear(conv_dim, d_model)
        
        # Positional encoding
        self.pos_conv = nn.Conv1d(
            d_model,
            d_model,
            kernel_size=128,
            padding=128 // 2,
            groups=16,
        )
        
        # Transformer
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=num_heads,
            dim_feedforward=d_model * 4,
            dropout=dropout,
            activation='gelu',
            batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers)
        
        # Output
        self.layer_norm = nn.LayerNorm(d_model)
        self.classifier = nn.Linear(d_model, vocab_size)
    
    def forward(
        self,
        features: Tensor,
        feature_lengths: Optional[Tensor] = None
    ) -> Tuple[Tensor, Tensor]:
        """Forward pass."""
        # Extract features
        if features.dim() == 2 or (features.dim() == 3 and features.size(1) == 1):
            x = self.feature_extractor(features)
        else:
            x = features
        
        # Project
        x = self.post_extract_proj(x)
        
        # Positional encoding
        x_conv = self.pos_conv(x.transpose(1, 2))
        x = x + x_conv.transpose(1, 2)
        
        # Encode
        x = self.encoder(x)
        x = self.layer_norm(x)
        
        # Classify
        logits = self.classifier(x)
        
        if feature_lengths is not None:
            output_lengths = feature_lengths
        else:
            output_lengths = torch.full((x.size(0),), x.size(1), dtype=torch.long)
        
        return logits, output_lengths


class Whisper(AcousticModel):
    """Whisper: Robust ASR model (simplified implementation)."""
    
    def __init__(
        self,
        vocab_size: int,
        d_model: int = 512,
        num_encoder_layers: int = 6,
        num_decoder_layers: int = 6,
        num_heads: int = 8,
        dropout: float = 0.0,
        max_source_positions: int = 1500,
        max_target_positions: int = 448,
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.d_model = d_model
        
        # Encoder (mel spectrogram -> hidden states)
        self.encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=d_model,
                nhead=num_heads,
                dim_feedforward=d_model * 4,
                dropout=dropout,
                activation='gelu',
                batch_first=True,
            ),
            num_encoder_layers,
        )
        
        # Decoder
        self.decoder = nn.TransformerDecoder(
            nn.TransformerDecoderLayer(
                d_model=d_model,
                nhead=num_heads,
                dim_feedforward=d_model * 4,
                dropout=dropout,
                activation='gelu',
                batch_first=True,
            ),
            num_decoder_layers,
        )
        
        # Embeddings
        self.decoder_embedding = nn.Embedding(vocab_size, d_model)
        
        # Output projection
        self.classifier = nn.Linear(d_model, vocab_size)
    
    def forward(
        self,
        features: Tensor,
        feature_lengths: Optional[Tensor] = None,
        decoder_input_ids: Optional[Tensor] = None,
    ) -> Tensor:
        """Forward pass."""
        # Encode
        encoder_out = self.encoder(features)
        
        if decoder_input_ids is None:
            # Inference mode
            return self._greedy_decode(encoder_out)
        
        # Decode
        decoder_emb = self.decoder_embedding(decoder_input_ids)
        decoder_out = self.decoder(decoder_emb, encoder_out)
        
        # Project to vocabulary
        logits = self.classifier(decoder_out)
        
        return logits
    
    def _greedy_decode(
        self,
        encoder_out: Tensor,
        max_len: int = 448,
    ) -> Tensor:
        """Greedy decoding."""
        batch_size = encoder_out.size(0)
        device = encoder_out.device
        
        # Start token (assume 50258 for <|startoftranscript|>)
        decoder_input = torch.full(
            (batch_size, 1),
            50258,
            dtype=torch.long,
            device=device,
        )
        
        for _ in range(max_len):
            decoder_emb = self.decoder_embedding(decoder_input)
            decoder_out = self.decoder(decoder_emb, encoder_out)
            logits = self.classifier(decoder_out[:, -1:, :])
            
            _, next_token = logits.max(dim=-1)
            decoder_input = torch.cat([decoder_input, next_token], dim=1)
            
            # Check for end token (assume 50257 for <|endoftext|>)
            if (next_token == 50257).all():
                break
        
        return decoder_input
