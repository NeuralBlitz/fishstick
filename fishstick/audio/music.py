"""
Comprehensive Music Generation Module

This module provides tools for symbolic music generation, audio music generation,
music representation, analysis, style transfer, accompaniment generation, and utilities.
"""

from __future__ import annotations

import abc
import random
import warnings
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import (
    Any,
    Callable,
    Dict,
    List,
    Literal,
    Optional,
    Protocol,
    Sequence,
    Tuple,
    Union,
)

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.utils.data import Dataset

# Type aliases
MidiSequence = List[Dict[str, Union[int, float, str]]]
PianoRoll = np.ndarray  # Shape: (num_time_steps, num_pitches)
AudioSignal = np.ndarray  # Shape: (num_samples,) or (num_channels, num_samples)


# =============================================================================
# Symbolic Music Generation
# =============================================================================


class MusicTransformer(nn.Module):
    """
    Transformer model for symbolic MIDI generation.

    Uses self-attention to model long-range dependencies in musical sequences.

    Args:
        vocab_size: Size of the MIDI event vocabulary
        d_model: Embedding dimension
        nhead: Number of attention heads
        num_layers: Number of transformer layers
        dim_feedforward: Feedforward network dimension
        dropout: Dropout rate
        max_seq_len: Maximum sequence length

    Example:
        >>> model = MusicTransformer(vocab_size=512, d_model=256)
        >>> output = model(input_tokens)  # (batch, seq_len, vocab_size)
    """

    def __init__(
        self,
        vocab_size: int = 512,
        d_model: int = 512,
        nhead: int = 8,
        num_layers: int = 12,
        dim_feedforward: int = 2048,
        dropout: float = 0.1,
        max_seq_len: int = 4096,
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.max_seq_len = max_seq_len

        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pos_encoding = MusicPositionalEncoding(d_model, dropout, max_seq_len)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers)

        self.output_proj = nn.Linear(d_model, vocab_size)
        self._init_weights()

    def _init_weights(self) -> None:
        """Initialize model weights."""
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, x: Tensor, mask: Optional[Tensor] = None) -> Tensor:
        """
        Forward pass through the transformer.

        Args:
            x: Input token indices, shape (batch_size, seq_len)
            mask: Optional attention mask

        Returns:
            Logits over vocabulary, shape (batch_size, seq_len, vocab_size)
        """
        x = self.embedding(x) * np.sqrt(self.d_model)
        x = self.pos_encoding(x)

        if mask is not None:
            x = self.transformer(x, src_key_padding_mask=mask)
        else:
            x = self.transformer(x)

        return self.output_proj(x)

    def generate(
        self,
        start_tokens: Tensor,
        max_length: int = 512,
        temperature: float = 1.0,
        top_k: Optional[int] = None,
        top_p: Optional[float] = None,
    ) -> Tensor:
        """
        Generate music autoregressively.

        Args:
            start_tokens: Initial token sequence, shape (batch_size, seq_len)
            max_length: Maximum tokens to generate
            temperature: Sampling temperature
            top_k: Top-k sampling parameter
            top_p: Nucleus sampling parameter

        Returns:
            Generated token sequence
        """
        self.eval()
        with torch.no_grad():
            generated = start_tokens.clone()

            for _ in range(max_length):
                if generated.size(1) >= self.max_seq_len:
                    break

                logits = self.forward(generated)[:, -1, :]
                logits = logits / temperature

                if top_k is not None:
                    indices_to_remove = (
                        logits < torch.topk(logits, top_k)[0][..., -1, None]
                    )
                    logits[indices_to_remove] = float("-inf")

                if top_p is not None:
                    sorted_logits, sorted_indices = torch.sort(logits, descending=True)
                    cumulative_probs = torch.cumsum(
                        F.softmax(sorted_logits, dim=-1), dim=-1
                    )
                    sorted_indices_to_remove = cumulative_probs > top_p
                    sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[
                        ..., :-1
                    ].clone()
                    sorted_indices_to_remove[..., 0] = 0
                    indices_to_remove = sorted_indices_to_remove.scatter(
                        1, sorted_indices, sorted_indices_to_remove
                    )
                    logits[indices_to_remove] = float("-inf")

                probs = F.softmax(logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)
                generated = torch.cat([generated, next_token], dim=1)

            return generated


class MusicPositionalEncoding(nn.Module):
    """Positional encoding for music sequences."""

    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-np.log(10000.0) / d_model))
        pe = torch.zeros(max_len, d_model)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe)

    def forward(self, x: Tensor) -> Tensor:
        """
        Args:
            x: Tensor of shape (batch_size, seq_len, d_model)
        """
        x = x + self.pe[: x.size(1)]
        return self.dropout(x)


class MuseGAN(nn.Module):
    """
    Multi-track music generation using GANs.

    Generates multiple instrument tracks simultaneously using a shared
    latent space and track-specific generators.

    Args:
        num_tracks: Number of instrument tracks to generate
        bar_length: Length of a bar in time steps
        pitch_range: Number of pitch classes
        latent_dim: Dimension of latent space
        hidden_dim: Hidden layer dimension

    Example:
        >>> model = MuseGAN(num_tracks=5, bar_length=96, pitch_range=84)
        >>> tracks = model.generate(batch_size=4)  # 5 tracks, each (4, 1, 84, 96)
    """

    def __init__(
        self,
        num_tracks: int = 5,
        bar_length: int = 96,
        pitch_range: int = 84,
        latent_dim: int = 128,
        hidden_dim: int = 256,
    ):
        super().__init__()
        self.num_tracks = num_tracks
        self.bar_length = bar_length
        self.pitch_range = pitch_range
        self.latent_dim = latent_dim

        # Shared temporal generator
        self.temporal_generator = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )

        # Track-specific generators
        self.track_generators = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(hidden_dim, hidden_dim * 2),
                    nn.ReLU(),
                    nn.Unflatten(1, (hidden_dim, 1, 1)),
                    nn.ConvTranspose2d(
                        hidden_dim,
                        hidden_dim // 2,
                        kernel_size=(4, 4),
                        stride=(2, 2),
                        padding=(1, 1),
                    ),
                    nn.BatchNorm2d(hidden_dim // 2),
                    nn.ReLU(),
                    nn.ConvTranspose2d(
                        hidden_dim // 2,
                        hidden_dim // 4,
                        kernel_size=(4, 4),
                        stride=(2, 2),
                        padding=(1, 1),
                    ),
                    nn.BatchNorm2d(hidden_dim // 4),
                    nn.ReLU(),
                    nn.ConvTranspose2d(
                        hidden_dim // 4,
                        1,
                        kernel_size=(4, 4),
                        stride=(2, 2),
                        padding=(1, 1),
                    ),
                    nn.Sigmoid(),
                )
                for _ in range(num_tracks)
            ]
        )

        # Reshape to piano roll dimensions
        self.pitch_upsample = (
            pitch_range // 8
        )  # Assuming generators output ~8 pitch bins
        self.time_upsample = bar_length // 8

        self.final_upsample = nn.ModuleList(
            [
                nn.Sequential(
                    nn.ConvTranspose2d(
                        1,
                        1,
                        kernel_size=(self.pitch_upsample, self.time_upsample),
                        stride=(self.pitch_upsample, self.time_upsample),
                    ),
                    nn.Sigmoid(),
                )
                for _ in range(num_tracks)
            ]
        )

    def forward(self, z: Tensor) -> List[Tensor]:
        """
        Generate multi-track piano rolls.

        Args:
            z: Latent vector, shape (batch_size, latent_dim)

        Returns:
            List of piano rolls, one per track
        """
        temporal_features = self.temporal_generator(z)

        tracks = []
        for i, gen in enumerate(self.track_generators):
            track = gen(temporal_features)
            track = self.final_upsample[i](track)
            tracks.append(track)

        return tracks

    def generate(self, batch_size: int = 1) -> List[Tensor]:
        """Generate random multi-track music."""
        z = torch.randn(batch_size, self.latent_dim)
        return self.forward(z)


class MuseGANDiscriminator(nn.Module):
    """Discriminator for MuseGAN multi-track music."""

    def __init__(
        self,
        num_tracks: int = 5,
        pitch_range: int = 84,
        bar_length: int = 96,
        hidden_dim: int = 64,
    ):
        super().__init__()
        self.num_tracks = num_tracks

        # Track-specific discriminators
        self.track_discriminators = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Conv2d(
                        1, hidden_dim, kernel_size=(4, 4), stride=(2, 2), padding=(1, 1)
                    ),
                    nn.LeakyReLU(0.2),
                    nn.Conv2d(
                        hidden_dim,
                        hidden_dim * 2,
                        kernel_size=(4, 4),
                        stride=(2, 2),
                        padding=(1, 1),
                    ),
                    nn.BatchNorm2d(hidden_dim * 2),
                    nn.LeakyReLU(0.2),
                    nn.Conv2d(
                        hidden_dim * 2,
                        hidden_dim * 4,
                        kernel_size=(4, 4),
                        stride=(2, 2),
                        padding=(1, 1),
                    ),
                    nn.BatchNorm2d(hidden_dim * 4),
                    nn.LeakyReLU(0.2),
                    nn.Flatten(),
                )
                for _ in range(num_tracks)
            ]
        )

        # Combined discriminator
        conv_output_size = hidden_dim * 4 * (pitch_range // 8) * (bar_length // 8)
        self.combiner = nn.Sequential(
            nn.Linear(conv_output_size * num_tracks, hidden_dim * 8),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden_dim * 8, 1),
            nn.Sigmoid(),
        )

    def forward(self, tracks: List[Tensor]) -> Tensor:
        """
        Discriminate multi-track music.

        Args:
            tracks: List of piano rolls, one per track

        Returns:
            Probability of being real music
        """
        features = []
        for i, track in enumerate(tracks):
            feat = self.track_discriminators[i](track)
            features.append(feat)

        combined = torch.cat(features, dim=1)
        return self.combiner(combined)


class LSTMComposer(nn.Module):
    """
    LSTM-based music composition model.

    Uses recurrent layers to model sequential dependencies in music.

    Args:
        vocab_size: Size of the note/event vocabulary
        embedding_dim: Embedding dimension
        hidden_dim: LSTM hidden dimension
        num_layers: Number of LSTM layers
        dropout: Dropout rate

    Example:
        >>> model = LSTMComposer(vocab_size=128, hidden_dim=256)
        >>> composition = model.generate(seed_sequence, length=256)
    """

    def __init__(
        self,
        vocab_size: int = 128,
        embedding_dim: int = 128,
        hidden_dim: int = 256,
        num_layers: int = 2,
        dropout: float = 0.3,
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(
            embedding_dim,
            hidden_dim,
            num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
        )
        self.dropout = nn.Dropout(dropout)
        self.output_proj = nn.Linear(hidden_dim, vocab_size)

    def forward(
        self, x: Tensor, hidden: Optional[Tuple[Tensor, Tensor]] = None
    ) -> Tuple[Tensor, Tuple[Tensor, Tensor]]:
        """
        Forward pass through LSTM.

        Args:
            x: Input tokens, shape (batch_size, seq_len)
            hidden: Optional hidden state

        Returns:
            logits: Output logits, shape (batch_size, seq_len, vocab_size)
            hidden: Final hidden state
        """
        embedded = self.dropout(self.embedding(x))
        lstm_out, hidden = self.lstm(embedded, hidden)
        lstm_out = self.dropout(lstm_out)
        logits = self.output_proj(lstm_out)
        return logits, hidden

    def generate(
        self,
        seed: Tensor,
        length: int = 256,
        temperature: float = 1.0,
    ) -> Tensor:
        """
        Generate music sequence.

        Args:
            seed: Initial sequence, shape (batch_size, seed_len)
            length: Number of tokens to generate
            temperature: Sampling temperature

        Returns:
            Generated sequence including seed
        """
        self.eval()
        device = next(self.parameters()).device
        generated = seed.clone().to(device)
        hidden = None

        with torch.no_grad():
            # Process seed to initialize hidden state
            _, hidden = self.forward(generated, hidden)

            for _ in range(length):
                logits, hidden = self.forward(generated[:, -1:], hidden)
                logits = logits[:, -1, :] / temperature
                probs = F.softmax(logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)
                generated = torch.cat([generated, next_token], dim=1)

        return generated


class MusicVAE(nn.Module):
    """
    Variational Autoencoder for music.

    Learns a compressed latent representation of music sequences
    that can be sampled and decoded to generate new music.

    Args:
        vocab_size: Size of the note vocabulary
        embedding_dim: Embedding dimension
        encoder_dim: Encoder hidden dimension
        decoder_dim: Decoder hidden dimension
        latent_dim: Latent space dimension
        sequence_length: Maximum sequence length

    Example:
        >>> model = MusicVAE(vocab_size=90, latent_dim=512)
        >>> reconstruction, mu, logvar = model(input_sequence)
    """

    def __init__(
        self,
        vocab_size: int = 90,
        embedding_dim: int = 256,
        encoder_dim: int = 512,
        decoder_dim: int = 512,
        latent_dim: int = 256,
        sequence_length: int = 256,
        num_encoder_layers: int = 2,
        num_decoder_layers: int = 2,
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.latent_dim = latent_dim
        self.sequence_length = sequence_length

        # Encoder
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.encoder_lstm = nn.LSTM(
            embedding_dim,
            encoder_dim,
            num_encoder_layers,
            batch_first=True,
            bidirectional=True,
        )

        # VAE parameters
        self.fc_mu = nn.Linear(encoder_dim * 2, latent_dim)
        self.fc_logvar = nn.Linear(encoder_dim * 2, latent_dim)

        # Decoder
        self.decoder_init = nn.Linear(latent_dim, decoder_dim)
        self.decoder_lstm = nn.LSTM(
            embedding_dim,
            decoder_dim,
            num_decoder_layers,
            batch_first=True,
        )
        self.output_proj = nn.Linear(decoder_dim, vocab_size)

    def encode(self, x: Tensor) -> Tuple[Tensor, Tensor]:
        """
        Encode sequence to latent parameters.

        Args:
            x: Input sequence, shape (batch_size, seq_len)

        Returns:
            mu: Mean of latent distribution
            logvar: Log variance of latent distribution
        """
        embedded = self.embedding(x)
        lstm_out, _ = self.encoder_lstm(embedded)

        # Use final hidden state
        final_hidden = lstm_out[:, -1, :]
        mu = self.fc_mu(final_hidden)
        logvar = self.fc_logvar(final_hidden)
        return mu, logvar

    def reparameterize(self, mu: Tensor, logvar: Tensor) -> Tensor:
        """Reparameterization trick."""
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(
        self,
        z: Tensor,
        target_seq: Optional[Tensor] = None,
        teacher_forcing_ratio: float = 0.5,
    ) -> Tensor:
        """
        Decode latent vector to sequence.

        Args:
            z: Latent vector, shape (batch_size, latent_dim)
            target_seq: Target sequence for teacher forcing
            teacher_forcing_ratio: Probability of using teacher forcing

        Returns:
            Output logits, shape (batch_size, seq_len, vocab_size)
        """
        batch_size = z.size(0)
        device = z.device

        # Initialize decoder hidden state
        hidden = (
            self.decoder_init(z)
            .unsqueeze(0)
            .repeat(self.decoder_lstm.num_layers, 1, 1),
            torch.zeros(
                self.decoder_lstm.num_layers,
                batch_size,
                self.decoder_lstm.hidden_size,
                device=device,
            ),
        )

        # Start token (assume 0 is start)
        input_token = torch.zeros(batch_size, 1, dtype=torch.long, device=device)
        outputs = []

        for t in range(self.sequence_length):
            embedded = self.embedding(input_token)
            lstm_out, hidden = self.decoder_lstm(embedded, hidden)
            logits = self.output_proj(lstm_out)
            outputs.append(logits)

            # Next input
            if target_seq is not None and torch.rand(1).item() < teacher_forcing_ratio:
                input_token = target_seq[:, t : t + 1]
            else:
                input_token = torch.argmax(logits, dim=-1)

        return torch.cat(outputs, dim=1)

    def forward(self, x: Tensor) -> Tuple[Tensor, Tensor, Tensor]:
        """
        Full forward pass through VAE.

        Returns:
            reconstruction: Reconstructed sequence logits
            mu: Latent mean
            logvar: Latent log variance
        """
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        reconstruction = self.decode(z, x)
        return reconstruction, mu, logvar

    def sample(self, num_samples: int = 1, device: str = "cpu") -> Tensor:
        """Sample new music from prior."""
        z = torch.randn(num_samples, self.latent_dim, device=device)
        return self.decode(z)

    def interpolate(self, x1: Tensor, x2: Tensor, num_steps: int = 8) -> Tensor:
        """
        Interpolate between two sequences in latent space.

        Args:
            x1: First sequence
            x2: Second sequence
            num_steps: Number of interpolation steps

        Returns:
            Interpolated sequences
        """
        self.eval()
        with torch.no_grad():
            mu1, _ = self.encode(x1)
            mu2, _ = self.encode(x2)

            interpolations = []
            for alpha in torch.linspace(0, 1, num_steps):
                z = (1 - alpha) * mu1 + alpha * mu2
                decoded = self.decode(z.unsqueeze(0))
                interpolations.append(decoded)

            return torch.cat(interpolations, dim=0)


# =============================================================================
# Audio Music Generation
# =============================================================================


class JukeboxModel(nn.Module):
    """
    Raw audio music generation model (simplified Jukebox-style).
    
    Uses hierarchical VQ-VAE with transformers for high-fidelity
    raw audio music generation at multiple time scales.
    
    Args:
        sample_rate: Audio sample rate
        hop_lengths: Hop lengths for each level (bottom to top)
        channels: Number of audio channels
        latent_dim: Latent dimension per level
        num_embeddings: VQ codebook size per level
        
    Example:
        >>> model = JukeboxModel(sample_rate=44100, hop_lengths=[512, 256, 128])
        >>> audio = model.generate(duration=10.0)  # Generate 10 seconds
    """
    
    def __init__(
        self,
        sample_rate: int = 22050,
        hop_lengths: List[int] = [512, 128, 32],
        channels: int = 1,
        latent_dim: int = 64,
        num_embeddings: List[int] = [2048, 2048, 2048],
        num_layers: List[int] = [4, 6, 12],
    ):
        super().__init__()
        self.sample_rate = sample_rate
        self.hop_lengths = hop_lengths
        self.num_levels = len(hop_lengths)
        
        # Create hierarchical VQ-VAE
        self.encoders = nn.ModuleList()
        self.decoders = nn.ModuleList()
        self.quantizers = nn.ModuleList()
        
        for i, (hop, num_emb, n_layers) in enumerate(zip(hop_lengths, num_embeddings, num_layers)):
            # Encoder
            encoder = nn.Sequential(
                nn.Conv1d(channels if i == 0 else latent_dim, latent_dim, hop, stride=hop//2, padding=hop//4),
                nn.ReLU(),
                nn.Conv1d(latent_dim, latent_dim, 3, padding=1),
                nn.ReLU(),
                nn.Conv1d(latent_dim, latent_dim, 1),
            )
            
            # Decoder
            decoder = nn.Sequential(
                nn.Conv1d(latent_dim, latent_dim, 3, padding=1),
                nn.ReLU(),
                nn.ConvTranspose1d(latent_dim, channels if i == 0 else latent_dim, hop, stride=hop//2, padding=hop//4),
            )
            
            # Quantizer
            from fishstick.audio.models import VectorQuantizer
            quantizer = VectorQuantizer(num_emb, latent_dim)
            
            self.encoders.append(encoder)
            self.decoders.append(decoder)
            self.quantizers.append(quantizer)
        
        # Prior models (transformers) for each level
        self.priors = nn.ModuleList([
            MusicTransformer(
                vocab_size=num_embeddings[i],
                d_model=512,
                nhead=8,
                num_layers=n_layers,
            )
            for i, n_layers in enumerate(num_layers)
        ])
    
    def encode(self, x: Tensor) -> List[Tuple[Tensor, Tensor]]:
        """
        Encode audio to hierarchical latent codes.
        
        Args:
            x: Audio waveform, shape (batch, channels, samples)
            
        Returns:
            List of (quantized, indices) for each level
        """
        codes = []
        current = x
        
        for encoder, quantizer in zip(self.encoders, self.quantizers):
            z = encoder(current)
            quant, indices, _ = quantizer(z.transpose(1, 2))
            codes.append((quant, indices))
            current = z
        
        return codes
    
    def decode(self, codes: List[Tensor]) -> Tensor:
        """
        Decode hierarchical codes to audio.
        
        Args:
            codes: List of quantized codes for each level
            
        Returns:
            Reconstructed audio
        """
        x = codes[-1].transpose(1, 2)
        
        for i in range(len(self.decoders) - 1, -1, -1):
            x = self.decoders[i](x)
        
        return x
    
    def generate(self, duration: float, temperature: float = 1.0) -> Tensor:
        """
        Generate raw audio music.
        
        Args:
            duration: Duration in seconds
            temperature: Sampling temperature
            
        Returns:
            Generated audio waveform
        """
        self.eval()
        num_samples = int(duration * self.sample_rate)
        
        with torch.no_grad():
            # Generate from top level to bottom
            generated_codes = []
            
            for level in range(len(self.priors) - 1, -1, -1):
                # Generate tokens for this level
                seq_len = num_samples // self.hop_lengths[level]
                start_token = torch.zeros(1, 1, dtype=torch.long)
                
                tokens = self.priors[level].generate(
                    start_token,
                    max_length=seq_len,
                    temperature=temperature,
                )
                
                # Get embeddings
                quant = self.quantizers[level].embedding(tokens[0])
                generated_codes.insert(0, quant.unsqueeze(0).transpose(1, 2))
            
            # Decode to audio
            audio = self.decode(generated_codes)
        
        return audio


class MusicDiffusion(nn.Module):
    """
    Diffusion model for music generation.
    
    Uses denoising diffusion probabilistic models (DDPM) to generate
    music spectrograms or raw audio.
    
    Args:
        input_channels: Number of input channels
        hidden_dim: Base hidden dimension
        num_res_blocks: Number of residual blocks
        attention_resolutions: Resolutions to apply attention
        dropout: Dropout rate
        num_diffusion_steps: Number of diffusion steps
        beta_schedule: Noise schedule type
        
    Example:
        >>> model = MusicDiffusion(input_channels=1, num_diffusion_steps=1000)
        >>> spectrogram = model.sample(shape=(1, 1, 256, 1024))
    """
    
    def __init__(
        self,
        input_channels: int = 1,
        hidden_dim: int = 128,
        num_res_blocks: int = 2,
        attention_resolutions: Tuple[int, ...] = (16, 8),
        dropout: float = 0.1,
        num_diffusion_steps: int = 1000,
        beta_schedule: str = 'linear',
    ):
        super().__init__()
        self.input_channels = input_channels
        self.num_diffusion_steps = num_diffusion_steps
        
        # Set up noise schedule
        if beta_schedule == 'linear':
            betas = torch.linspace(1e-4, 0.02, num_diffusion_steps)
        elif beta_schedule == 'cosine':
            betas = self._cosine_beta_schedule(num_diffusion_steps)
        else:
            raise ValueError(f"Unknown beta schedule: {beta_schedule}")
        
        alphas = 1.0 - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)
        
        self.register_buffer('betas', betas)
        self.register_buffer('alphas', alphas)
        self.register_buffer('alphas_cumprod', alphas_cumprod)
        self.register_buffer('sqrt_alphas_cumprod', torch.sqrt(alphas_cumprod))
        self.register_buffer('sqrt_one_minus_alphas_cumprod', torch.sqrt(1.0 - alphas_cumprod))
        
        # U-Net denoising model
        self.denoiser = UNetDenoiser(
            input_channels=input_channels,
            hidden_dim=hidden_dim,
            num_res_blocks=num_res_blocks,
            attention_resolutions=attention_resolutions,
            dropout=dropout,
            time_emb_dim=hidden_dim * 4,
        )
    
    def _cosine_beta_schedule(self, timesteps: int, s: float = 0.008) -> Tensor:
        """Cosine schedule as in improved DDPM."""
        steps = timesteps + 1
        x = torch.linspace(0, timesteps, steps)
        alphas_cumprod = torch.cos(((x / timesteps) + s) / (1 + s) * np.pi * 0.5) ** 2
        alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
        betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
        return torch.clip(betas, 0.0001, 0.9999)
    
    def forward_diffusion(self, x_0: Tensor, t: Tensor, noise: Optional[Tensor] = None) -> Tuple[Tensor, Tensor]:
        """
        Apply forward diffusion process.
        
        Args:
            x_0: Original data
            t: Timesteps
            noise: Optional noise to use
            
        Returns:
            x_t: Noised data
            noise: The noise that was added
        """
        if noise is None:
            noise = torch.randn_like(x_0)
        
        sqrt_alpha_prod = self.sqrt_alphas_cumprod[t].view(-1, 1, 1, 1)
        sqrt_one_minus_alpha_prod = self.sqrt_one_minus_alphas_cumprod[t].view(-1, 1, 1, 1)
        
        x_t = sqrt_alpha_prod * x_0 + sqrt_one_minus_alpha_prod * noise
        return x_t, noise
    
    def forward(self, x: Tensor) -> Tensor:
        """
        Training forward pass.
        
        Args:
            x: Clean data
            
        Returns:
            MSE loss between predicted and actual noise
        """
        batch_size = x.size(0)
        t = torch.randint(0, self.num_diffusion_steps, (batch_size,), device=x.device)
        
        x_t, noise = self.forward_diffusion(x, t)
        predicted_noise = self.denoiser(x_t, t)
        
        return F.mse_loss(predicted_noise, noise)
    
    @torch.no_grad()
    def sample(
        self,
        shape: Tuple[int, ...],
        num_inference_steps: Optional[int] = None,
    ) -> Tensor:
        """
        Generate samples using DDPM sampling.
        
        Args:
            shape: Shape of output tensor
            num_inference_steps: Number of denoising steps (defaults to num_diffusion_steps)
            
        Returns:
            Generated samples
        """
        self.eval()
        device = next(self.parameters()).device
        
        if num_inference_steps is None:
            num_inference_steps = self.num_diffusion_steps
        
        # Start from pure noise
        x = torch.randn(shape, device=device)
        
        # Reverse diffusion process
        for i in reversed(range(num_inference_steps)):
            t = torch.full((shape[0],), i, device=device, dtype=torch.long)
            
            # Predict noise
            predicted_noise = self.denoiser(x, t)
            
            # Compute x_{t-1}
            alpha = self.alphas[t].view(-1, 1, 1, 1)
            alpha_cumprod = self.alphas_cumprod[t].view(-1, 1, 1, 1)
            beta = self.betas[t].view(-1, 1, 1, 1)
            
            # Mean of p(x_{t-1} | x_t)
            pred_x0 = (x - torch.sqrt(1 - alpha_cumprod) * predicted_noise) / torch.sqrt(alpha_cumprod)
            pred_x0 = torch.clamp(pred_x0, -1, 1)
            
            if i > 0:
                noise = torch.randn_like(x)
                alpha_cumprod_prev = self.alphas_cumprod[t - 1].view(-1, 1, 1, 1)
                x = (
                    torch.sqrt(alpha_cumprod_prev) * pred_x0 +
                    torch.sqrt(1 - alpha_cumprod_prev) * predicted_noise +
                    torch.sqrt(beta) * noise
                )
            else:
                x = pred_x0
        
        return x


class UNetDenoiser(nn.Module):
    """U-Net architecture for denoising in diffusion models."""
    
    def __init__(
        self,
        input_channels: int,
        hidden_dim: int,
        num_res_blocks: int,
        attention_resolutions: Tuple[int, ...],
        dropout: float,
        time_emb_dim: int,
    ):
        super().__init__()
        self.time_mlp = nn.Sequential(
            nn.Linear(hidden_dim, time_emb_dim),
            nn.SiLU(),
            nn.Linear(time_emb_dim, time_emb_dim),
        )
        
        # Encoder
        self.input_conv = nn.Conv2d(input_channels, hidden_dim, 3, padding=1)
        
        self.encoder_blocks = nn.ModuleList([
            ResnetBlock(hidden_dim, hidden_dim, time_emb_dim, dropout)
            for _ in range(num_res_blocks)
        ])
        
        self.downsample = nn.Conv2d(hidden_dim, hidden_dim * 2, 3, stride=2, padding=1)
        
        # Middle with attention
        self.middle_blocks = nn.ModuleList([
            ResnetBlock(hidden_dim * 2, hidden_dim * 2, time_emb_dim, dropout),
            AttentionBlock(hidden_dim * 2),
            ResnetBlock(hidden_dim * 2, hidden_dim * 2, time_emb_dim, dropout),
        ])
        
        # Decoder
        self.upsample = nn.ConvTranspose2d(hidden_dim * 2, hidden_dim, 4, stride=2, padding=1)
        
        self.decoder_blocks = nn.ModuleList([
            ResnetBlock(hidden_dim * 2, hidden_dim, time_emb_dim, dropout)
            for _ in range(num_res_blocks)
        ])
        
        self.output_conv = nn.Conv2d(hidden_dim, input_channels, 1)
    
    def forward(self, x: Tensor, t: Tensor) -> Tensor:
        """
        Forward pass of denoiser.
        
        Args:
            x: Noisy input, shape (batch, channels, height, width)
            t: Timestep indices, shape (batch,)
            
        Returns:
            Predicted noise
        """
        # Time embedding
        t_emb = self._timestep_embedding(t, x.size(1) * 4)
        t_emb = self.time_mlp(t_emb)
        
        # Encoder
        h = self.input_conv(x)
        skips = [h]
        
        for block in self.encoder_blocks:
            h = block(h, t_emb)
            skips.append(h)
        
        h = self.downsample(h)
        
        # Middle
        for block in self.middle_blocks:
            if isinstance(block, AttentionBlock):
                h = block(h)
            else:
                h = block(h, t_emb)
        
        # Decoder
        h = self.upsample(h)
        
        for block in self.decoder_blocks:
            h = torch.cat([h, skips.pop()], dim=1)
            h = block(h, t_emb)
        
        return self.output_conv(h)
    
    def _timestep_embedding(self, timesteps: Tensor, dim: int, max_period: int = 10000) -> Tensor:
        """Create sinusoidal timestep embeddings."""
        half = dim // 2
        freqs = torch.exp(
            -np.log(max_period) * torch.arange(0, half, dtype=torch.float32, device=timesteps.device) / half
        )
        args = timesteps[:, None].float() * freqs[None]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        return embedding


class ResnetBlock(nn.Module):
    """Residual block with time conditioning."""
    
    def __init__(self, in_channels: int, out_channels: int, time_emb_dim: int, dropout: float):
        super().__init__()
        self.norm1 = nn.GroupNorm(32, in_channels)
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, padding=1)
        
        self.time_proj = nn.Linear(time_emb_dim, out_channels)
        
        self.norm2 = nn.GroupNorm(32, out_channels)
        self.dropout = nn.Dropout(dropout)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1)
        
        if in_channels != out_channels:
            self.shortcut = nn.Conv2d(in_channels, out_channels, 1)
        else:
            self.shortcut = nn.Identity()
    
    def forward(self, x: Tensor, t_emb: Tensor) -> Tensor:
        h = self.norm1(x)
        h = F.silu(h)
        h = self.conv1(h)
        
        # Add time embedding
        h = h + self.time_proj(F.silu(t_emb))[:, :, None, None]
        
        h = self.norm2(h)
        h = F.silu(h)
        h = self.dropout(h)
        h = self.conv2(h)
        
        return h + self.shortcut(x)


class AttentionBlock(nn.Module):
    """Self-attention block."""
    
    def __init__(self, channels: int):
        super().__init__()
        self.norm = nn.GroupNorm(32, channels)
        self.qkv = nn.Conv2d(channels, channels * 3, 1)
        self.proj = nn.Conv2d(channels, channels, 1)
    
    def forward(self, x: Tensor) -> Tensor:
        B, C, H, W = x.shape
        h = self.norm(x)
        qkv = self.qkv(h)
        q, k, v = qkv.chunk(3, dim=1)
        
        # Reshape for attention
        q = q.view(B, C, H * W).transpose(1, 2)
        k = k.view(B, C, H * W).transpose(1, 2)
        v = v.view(B, C, H * W).transpose(1, 2)
        
        # Attention
        scale = 1.0 / np.sqrt(C)
        attn = torch.softmax(q @ k.transpose(1, 2) * scale, dim=-1)
        h = attn @ v
        
        h = h.transpose(1, 2).view(B, C, H, W)
        h = self.proj(h)
        
        return x + h


class WaveNetMusic(nn.Module):
    """
    Autoregressive audio generation using WaveNet architecture.
    
    Generates raw audio samples one at a time using dilated causal
    convolutions with residual connections.
    
    Args:
        num_channels: Number of audio channels
        num_layers: Number of WaveNet layers
        num_blocks: Number of residual blocks
        residual_channels: Residual channel dimension
        dilation_channels: Dilation channel dimension
        skip_channels: Skip connection channel dimension
        quantization_levels: Number of mu-law quantization levels
        
    Example:
        >>> model = WaveNetMusic(num_layers=10, num_blocks=2)
        >>> audio = model.fast_generate(length=16000)  # 1 second at 16kHz
    """
    
    def __init__(
        self,
        num_channels: int = 1,
        num_layers: int = 10,
        num_blocks: int = 2,
        residual_channels: int = 512,
        dilation_channels: int = 512,
        skip_channels: int = 256,
        quantization_levels: int = 256,
    ):
        super().__init__()
        self.num_layers = num_layers
        self.num_blocks = num_blocks
        self.quantization_levels = quantization_levels
        
        # Input embedding
        self.input_conv = nn.Conv1d(1, residual_channels, 1)
        
        # Dilated convolution layers
        self.layers = nn.ModuleList()
        for b in range(num_blocks):
            for i in range(num_layers):
                dilation = 2 ** i
                self.layers.append(
                    WaveNetLayer(
                        residual_channels=residual_channels,
                        dilation_channels=dilation_channels,
                        skip_channels=skip_channels,
                        kernel_size=2,
                        dilation=dilation,
                    )
                )
        
        # Output layers
        self.output_conv = nn.Sequential(
            nn.ReLU(),
            nn.Conv1d(skip_channels, skip_channels, 1),
            nn.ReLU(),
            nn.Conv1d(skip_channels, quantization_levels, 1),
        )
    
    def forward(self, x: Tensor) -> Tensor:
        """
        Forward pass through WaveNet.
        
        Args:
            x: Input audio, shape (batch, 1, samples)
            
        Returns:
            Logits over quantization levels, shape (batch, quantization_levels, samples)
        """
        x = self.input_conv(x)
        
        skip_connections = []
        for layer in self.layers:
            x, skip = layer(x)
            skip_connections.append(skip)
        
        # Combine skip connections
        x = sum(skip_connections)
        x = self.output_conv(x)
        
        return x
    
    def generate(self, num_samples: int, temperature: float = 1.0) -> Tensor:
        """
        Generate audio sample by sample (slow).
        
        Args:
            num_samples: Number of samples to generate
            temperature: Sampling temperature
            
        Returns:
            Generated audio
        """
        self.eval()
        device = next(self.parameters()).device
        
        # Start with zeros
        generated = torch.zeros(1, 1, 1, device=device)
        
        with torch.no_grad():
            for _ in range(num_samples):
                logits = self.forward(generated)[:, :, -1:]
                logits = logits / temperature
                probs = F.softmax(logits, dim=1)
                sample = torch.multinomial(probs.squeeze(-1), num_samples=1)
                sample = (sample.float() / self.quantization_levels) * 2 - 1  # Dequantize
                generated = torch.cat([generated, sample.unsqueeze(1)], dim=2)
        
        return generated[:, :, 1:]  # Remove initial zero


class WaveNetLayer(nn.Module):
    """Single WaveNet layer with gated activation."""
    
    def __init__(
        self,
        residual_channels: int,
        dilation_channels: int,
        skip_channels: int,
        kernel_size: int,
        dilation: int,
    ):
        super().__init__()
        self.dilation = dilation
        
        self.conv = nn.Conv1d(
            residual_channels,
            dilation_channels * 2,  # For gate and filter
            kernel_size,
            dilation=dilation,
        )
        
        self.residual_conv = nn.Conv1d(dilation_channels, residual_channels, 1)
        self.skip_conv = nn.Conv1d(dilation_channels, skip_channels, 1)
    
    def forward(self, x: Tensor) -> Tuple[Tensor, Tensor]:
        """Forward pass of WaveNet layer."""
        # Causal padding
        padding = (self.conv.kernel_size[0] - 1) * self.dilation
        x_padded = F.pad(x, (padding, 0))
        
        conv_out = self.conv(x_padded)
        filter_out, gate_out = conv_out.chunk(2, dim=1)
        
        # Gated activation
        activation = torch.tanh(filter_out) * torch.sigmoid(gate_out)
        
        # Residual and skip connections
        residual = self.residual_conv(activation)
        skip = self.skip_conv(activation)
        
        return (x + residual)[:, :, -activation.size(2):], skip


class GANSynth(nn.Module):
    """
    GAN for generating musical notes.
    
    Generates high-quality musical instrument notes with
    pitch and velocity conditioning.
    
    Args:
        latent_dim: Latent space dimension
        num_pitches: Number of pitch classes to generate
        sample_rate: Audio sample rate
        audio_length: Length of generated audio in samples
        
    Example:
        >>> model = GANSynth(num_pitches=88, sample_rate=16000, audio_length=64000)
        >>> audio = model.generate(pitch=60, velocity=100)
    """
    
    def __init__(
        self,
        latent_dim: int = 256,
        num_pitches: int = 88,
        num_velocities: int = 128,
        sample_rate: int = 16000,
        audio_length: int = 64000,  # 4 seconds at 16kHz
    ):
        super().__init__()
        self.latent_dim = latent_dim
        self.num_pitches = num_pitches
        self.num_velocities = num_velocities
        self.sample_rate = sample_rate
        self.audio_length = audio_length
        
        # Pitch and velocity embeddings
        self.pitch_embed = nn.Embedding(num_pitches, 64)
        self.velocity_embed = nn.Embedding(num_velocities, 64)
        
        # Generator
        self.generator = nn.Sequential(
            # Initial projection
            nn.Linear(latent_dim + 128, 1024),
            nn.ReLU(),
            
            # Upsample to audio length
            nn.Unflatten(1, (256, 4)),
            
            # ConvTranspose layers to reach audio_length
            nn.ConvTranspose1d(256, 256, kernel_size=16, stride=4, padding=6),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            
            nn.ConvTranspose1d(256, 128, kernel_size=16, stride=4, padding=6),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            
            nn.ConvTranspose1d(128, 64, kernel_size=16, stride=4, padding=6),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            
            nn.ConvTranspose1d(64, 32, kernel_size=16, stride=4, padding=6),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            
            # Final conv to single channel
            nn.Conv1d(32, 1, kernel_size=7, padding=3),
            nn.Tanh(),
        )
        
        # Check output size and adjust if needed
        with torch.no_grad():
            test_input = torch.randn(1, latent_dim + 128)
            test_output = self.generator(test_input)
            if test_output.size(2) != audio_length:
                self.final_upsample = nn.ConvTranspose1d(1, 1, kernel_size=audio_length - test_output.size(2) + 1, stride=1)
            else:
                self.final_upsample = None
    
    def forward(self, z: Tensor, pitch: Tensor, velocity: Tensor) -> Tensor:
        """
        Generate audio conditioned on pitch and velocity.
        
        Args:
            z: Latent vector, shape (batch, latent_dim)
            pitch: Pitch class indices, shape (batch,)
            velocity: Velocity values, shape (batch,)
            
        Returns:
            Generated audio, shape (batch, 1, audio_length)
        """
        pitch_emb = self.pitch_embed(pitch)
        velocity_emb = self.velocity_embed(velocity)
        
        z_cond = torch.cat([z, pitch_emb, velocity_emb], dim=1)
        audio = self.generator(z_cond)
        
        if self.final_upsample is not None:
            audio = self.final_upsample(audio)
        
        return audio[:, :, :self.audio_length]
    
    def generate(
        self,
        pitch: int = 60,
        velocity: int = 100,
        num_samples: int = 1,
    ) -> Tensor:
        """Generate audio for specific pitch and velocity."""
        z = torch.randn(num_samples, self.latent_dim)
        pitch_tensor = torch.full((num_samples,), pitch, dtype=torch.long)
        velocity_tensor = torch.full((num_samples,), velocity, dtype=torch.long)
        return self.forward(z, pitch_tensor, velocity_tensor)
