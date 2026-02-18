"""
Speaker Separation

Implementation of speaker extraction and separation models using
speaker embeddings and target speaker conditioning.
"""

from typing import Optional, List, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F
from fishstick.audio_separation.base import SeparationModel, SeparationResult, STFT


class SpeakerEncoder(nn.Module):
    """Speaker Encoder for extracting speaker embeddings.

    Processes enrollment audio to extract speaker-specific embeddings
    that can be used for speaker-conditioned separation.
    """

    def __init__(
        self,
        input_dim: int = 80,
        hidden_dim: int = 256,
        embedding_dim: int = 192,
        n_layers: int = 3,
    ):
        super().__init__()

        self.lstm = nn.LSTM(
            input_dim,
            hidden_dim,
            n_layers,
            batch_first=True,
            bidirectional=True,
        )

        self.embedding = nn.Sequential(
            nn.Linear(hidden_dim * 2, embedding_dim),
            nn.ReLU(),
            nn.Linear(embedding_dim, embedding_dim),
        )

    def forward(
        self,
        features: torch.Tensor,
        lengths: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Extract speaker embedding.

        Args:
            features: Log-mel features of shape (batch, time, freq)
            lengths: Optional sequence lengths

        Returns:
            Speaker embedding of shape (batch, embedding_dim)
        """
        if features.dim() == 3:
            features = features.transpose(1, 2)

        output, (hidden, _) = self.lstm(features)

        hidden_concat = torch.cat([hidden[-2], hidden[-1]], dim=1)

        embedding = self.embedding(hidden_concat)

        embedding = F.normalize(embedding, p=2, dim=1)

        return embedding

    def compute_similarity(
        self,
        embedding1: torch.Tensor,
        embedding2: torch.Tensor,
    ) -> torch.Tensor:
        """Compute cosine similarity between two embeddings."""
        return torch.sum(embedding1 * embedding2, dim=1)


class SpeakerBeam(SeparationModel):
    """SpeakerBeam for speaker-conditioned source separation.

    Uses a reference speaker utterance to guide separation of that
    speaker's voice from a mixture.

    Reference:
        SpeakerBeam: Target Speaker Driven Speech Separation
    """

    def __init__(
        self,
        n_sources: int = 2,
        sample_rate: int = 16000,
        n_fft: int = 512,
        hop_length: int = 128,
        hidden_dim: int = 256,
        embedding_dim: int = 192,
    ):
        super().__init__(n_sources, sample_rate, n_fft, hop_length)

        self.stft = STFT(n_fft, hop_length)

        self.speaker_encoder = SpeakerEncoder(embedding_dim=embedding_dim)

        self.mask_estimator = nn.Sequential(
            nn.Linear(hidden_dim + embedding_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, n_fft // 2 + 1),
            nn.Sigmoid(),
        )

        self.feature_extractor = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, 3, padding=1),
            nn.ReLU(),
        )

    def forward(
        self,
        mixture: torch.Tensor,
        reference: Optional[torch.Tensor] = None,
    ) -> SeparationResult:
        """Separate speaker from mixture using reference.

        Args:
            mixture: Mixed audio
            reference: Reference audio of target speaker

        Returns:
            SeparationResult with separated sources
        """
        mix_stft = self.stft(mixture)
        if mix_stft.dim() == 4:
            mix_stft = mix_stft.squeeze(2)

        mag = torch.abs(mix_stft)
        phase = torch.angle(mix_stft)

        log_mag = torch.log(mag + 1e-8)

        if reference is not None:
            ref_features = self._extract_features(reference)
            speaker_emb = self.speaker_encoder(ref_features)
        else:
            speaker_emb = torch.zeros(mag.shape[0], 192, device=mag.device)

        mag_flat = mag.permute(0, 2, 1)

        combined = torch.cat(
            [mag_flat, speaker_emb.unsqueeze(1).expand(-1, mag_flat.shape[1], -1)],
            dim=-1,
        )

        masks = self.mask_estimator(combined)

        masks = masks.permute(0, 2, 1).unsqueeze(-1)

        target_mag = mag * masks
        target_stft = torch.complex(
            target_mag * torch.cos(phase), target_mag * torch.sin(phase)
        )

        target_wav = self.stft.inverse(target_stft)

        others_mag = mag * (1 - masks)
        others_stft = torch.complex(
            others_mag * torch.cos(phase), others_mag * torch.sin(phase)
        )
        others_wav = self.stft.inverse(others_stft)

        sources = torch.stack([target_wav, others_wav])

        return SeparationResult(
            sources=sources,
            source_names=["target", "interference"],
            masks=masks,
        )

    def _extract_features(self, audio: torch.Tensor) -> torch.Tensor:
        """Extract mel spectrogram features from audio."""
        if audio.dim() == 2:
            audio = audio.unsqueeze(1)

        import librosa

        features = []
        for i in range(audio.shape[0]):
            wav = audio[i, 0].cpu().numpy()
            mel = librosa.feature.melspectrogram(y=wav, sr=self.sample_rate, n_mels=80)
            features.append(torch.from_numpy(mel.T).float().to(audio.device))

        return torch.stack(features)

    def estimate_sources(
        self,
        mixture: torch.Tensor,
        reference: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Estimate sources from mixture."""
        result = self.forward(mixture, reference)
        return result.sources


class TarVAE(nn.Module):
    """Target Speaker Variational Autoencoder (Tar-VAE).

    A variational approach to speaker extraction that models the
    target speaker as a latent variable.

    Reference:
        Target Speaker Voice Activation with Complex Attention
    """

    def __init__(
        self,
        latent_dim: int = 64,
        n_fft: int = 512,
        hop_length: int = 128,
        hidden_dim: int = 128,
    ):
        super().__init__()

        self.latent_dim = latent_dim
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
        )

        self.fc_mu = nn.Linear(128 * freq_bins // 4, latent_dim)
        self.fc_var = nn.Linear(128 * freq_bins // 4, latent_dim)

        self.decoder = nn.Sequential(
            nn.Linear(latent_dim + freq_bins, 128 * freq_bins // 4),
            nn.LeakyReLU(0.2),
            nn.ConvTranspose2d(128, 64, 4, stride=2, padding=1),
            nn.LeakyReLU(0.2),
            nn.ConvTranspose2d(64, 32, 4, stride=2, padding=1),
            nn.LeakyReLU(0.2),
            nn.Conv2d(32, 1, 3, padding=1),
            nn.Sigmoid(),
        )

    def encode(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Encode to latent distribution."""
        h = self.encoder(x)
        h = h.reshape(h.shape[0], -1)

        mu = self.fc_mu(h)
        log_var = self.fc_var(h)

        return mu, log_var

    def reparameterize(self, mu: torch.Tensor, log_var: torch.Tensor) -> torch.Tensor:
        """Reparameterization trick."""
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z: torch.Tensor, condition: torch.Tensor) -> torch.Tensor:
        """Decode from latent to mask."""
        combined = torch.cat([z, condition], dim=1)

        h = combined.reshape(combined.shape[0], 128, self.n_fft // 4, 1)

        mask = self.decoder(h)

        return mask

    def forward(
        self,
        mixture: torch.Tensor,
        reference: Optional[torch.Tensor] = None,
    ) -> SeparationResult:
        """Extract target speaker using Tar-VAE."""
        mix_stft = self.stft(mixture)
        if mix_stft.dim() == 4:
            mix_stft = mix_stft.squeeze(2)

        mag = torch.abs(mix_stft).unsqueeze(1)

        if reference is not None:
            ref_mag = self._get_reference_features(reference)
        else:
            ref_mag = torch.zeros_like(mag)

        combined = torch.cat([mag, ref_mag], dim=1)

        mu, log_var = self.encode(combined)

        z = self.reparameterize(mu, log_var)

        mask = self.decode(z, mag.squeeze(1))

        mask = mask.squeeze(1)

        target_mag = mag.squeeze(1) * mask

        target_stft = torch.complex(
            target_mag * torch.cos(torch.angle(mix_stft)),
            target_mag * torch.sin(torch.angle(mix_stft)),
        )

        target_wav = self.stft.inverse(target_stft)

        return SeparationResult(sources=target_wav.unsqueeze(0))

    def _get_reference_features(self, reference: torch.Tensor) -> torch.Tensor:
        """Get reference features."""
        ref_stft = self.stft(reference)
        if ref_stft.dim() == 4:
            ref_stft = ref_stft.squeeze(2)
        return torch.abs(ref_stft).unsqueeze(1)


class SpeakerVerification(nn.Module):
    """Speaker Verification module for checking speaker identity.

    Can be used to verify if an extracted voice matches the target speaker.
    """

    def __init__(
        self,
        embedding_dim: int = 192,
        hidden_dim: int = 128,
    ):
        super().__init__()

        self.speaker_encoder = SpeakerEncoder(embedding_dim=embedding_dim)

        self.classifier = nn.Sequential(
            nn.Linear(embedding_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid(),
        )

    def forward(
        self,
        audio1: torch.Tensor,
        audio2: torch.Tensor,
    ) -> torch.Tensor:
        """Verify if two audio samples are from the same speaker.

        Args:
            audio1: First audio sample
            audio2: Second audio sample

        Returns:
            Similarity score between 0 and 1
        """
        features1 = self._extract_features(audio1)
        features2 = self._extract_features(audio2)

        emb1 = self.speaker_encoder(features1)
        emb2 = self.speaker_encoder(features2)

        combined = torch.cat([emb1, emb2], dim=1)

        similarity = self.classifier(combined)

        return similarity

    def _extract_features(self, audio: torch.Tensor) -> torch.Tensor:
        """Extract mel features from audio."""
        if audio.dim() == 2:
            audio = audio.unsqueeze(1)

        import librosa

        features = []
        for i in range(audio.shape[0]):
            wav = audio[i, 0].cpu().numpy()
            mel = librosa.feature.melspectrogram(y=wav, sr=16000, n_mels=80)
            features.append(torch.from_numpy(mel.T).float().to(audio.device))

        return torch.stack(features)


class SpeakerDiarization(nn.Module):
    """Speaker Diarization module for multi-speaker scenarios.

    Identifies and segments different speakers in an audio recording.
    """

    def __init__(
        self,
        n_speakers: int = 2,
        embedding_dim: int = 192,
        hidden_dim: int = 256,
    ):
        super().__init__()

        self.n_speakers = n_speakers
        self.embedding_dim = embedding_dim

        self.speaker_encoder = SpeakerEncoder(embedding_dim=embedding_dim)

        self.segment_classifier = nn.Sequential(
            nn.Linear(embedding_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, n_speakers),
        )

    def forward(
        self,
        audio: torch.Tensor,
        segment_length: int = 1600,
    ) -> torch.Tensor:
        """Perform speaker diarization.

        Args:
            audio: Input audio
            segment_length: Length of each segment for classification

        Returns:
            Speaker labels for each segment
        """
        if audio.dim() == 2:
            audio = audio.unsqueeze(1)

        batch, channels, length = audio.shape

        n_segments = length // segment_length

        segments = audio[:, :, : n_segments * segment_length]
        segments = segments.reshape(batch, n_segments, channels, segment_length)

        labels = []
        for i in range(n_segments):
            seg = segments[:, i]
            features = self._extract_features(seg)
            emb = self.speaker_encoder(features)
            label = self.segment_classifier(emb)
            labels.append(label)

        return torch.stack(labels, dim=1)

    def _extract_features(self, audio: torch.Tensor) -> torch.Tensor:
        """Extract mel features."""
        if audio.dim() == 3:
            audio = audio.squeeze(1)

        import librosa

        wav = audio[0].cpu().numpy()
        mel = librosa.feature.melspectrogram(y=wav, sr=16000, n_mels=80)
        return torch.from_numpy(mel.T).float().to(audio.device).unsqueeze(0)


class VoiceFilter(nn.Module):
    """VoiceFilter: Speaker-conditioned voice separation.

    Uses speaker embeddings to filter out unwanted speakers.

    Reference:
        VoiceFilter: Speaker-conditioned voice separation
    """

    def __init__(
        self,
        n_fft: int = 512,
        hop_length: int = 128,
        embedding_dim: int = 256,
    ):
        super().__init__()

        self.n_fft = n_fft
        self.hop_length = hop_length
        self.embedding_dim = embedding_dim

        self.stft = STFT(n_fft, hop_length)

        freq_bins = n_fft // 2 + 1

        self.speaker_encoder = SpeakerEncoder(embedding_dim=embedding_dim)

        self.attention = nn.MultiheadAttention(
            embed_dim=freq_bins,
            num_heads=8,
            batch_first=True,
        )

        self.mask_predictor = nn.Sequential(
            nn.Linear(freq_bins + embedding_dim, freq_bins),
            nn.ReLU(),
            nn.Linear(freq_bins, freq_bins),
            nn.Sigmoid(),
        )

    def forward(
        self,
        mixture: torch.Tensor,
        reference: torch.Tensor,
    ) -> torch.Tensor:
        """Filter target speaker voice from mixture.

        Args:
            mixture: Mixed audio
            reference: Reference audio of target speaker

        Returns:
            Extracted target speaker audio
        """
        mix_stft = self.stft(mixture)
        if mix_stft.dim() == 4:
            mix_stft = mix_stft.squeeze(2)

        mag = torch.abs(mix_stft)
        phase = torch.angle(mix_stft)

        ref_features = self._extract_features(reference)
        speaker_emb = self.speaker_encoder(ref_features)

        mag_t = mag.permute(0, 2, 1)

        attn_out, _ = self.attention(mag_t, mag_t, mag_t)

        combined = torch.cat(
            [attn_out, speaker_emb.unsqueeze(1).expand(-1, attn_out.shape[1], -1)],
            dim=-1,
        )

        mask = self.mask_predictor(combined)

        mask = mask.permute(0, 2, 1)

        enhanced_mag = mag * mask

        enhanced_stft = torch.complex(
            enhanced_mag * torch.cos(phase), enhanced_mag * torch.sin(phase)
        )

        enhanced_wav = self.stft.inverse(enhanced_stft)

        return enhanced_wav

    def _extract_features(self, audio: torch.Tensor) -> torch.Tensor:
        """Extract mel features."""
        import librosa

        if audio.dim() == 3:
            audio = audio.squeeze(1)

        wav = audio[0].cpu().numpy()
        mel = librosa.feature.melspectrogram(y=wav, sr=16000, n_mels=80)
        return torch.from_numpy(mel.T).float().to(audio.device).unsqueeze(0)
