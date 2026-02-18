"""
Speaker Recognition and Verification

Speaker identification and verification modules:
- Speaker embeddings extraction (x-vectors, d-vectors)
- Speaker verification
- Speaker identification
- Enrollment and comparison utilities
"""

from typing import Optional, List, Dict, Tuple, Union
from dataclasses import dataclass, field

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


@dataclass
class SpeakerConfig:
    """Configuration for speaker recognition."""

    sample_rate: int = 16000
    embedding_dim: int = 256
    hidden_dim: int = 512
    num_layers: int = 3
    dropout: float = 0.2
    num_speakers: int = 0
    margin: float = 0.3
    num_mels: int = 40


class SpeakerEncoder(nn.Module):
    """Speaker embedding encoder using TDNN-style architecture.

    Extracts speaker embeddings (x-vectors) from audio features.
    """

    def __init__(
        self,
        input_dim: int = 40,
        embedding_dim: int = 256,
        hidden_dim: int = 512,
        num_layers: int = 3,
        dropout: float = 0.2,
    ):
        super().__init__()

        self.input_dim = input_dim
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim

        self.frame_layer = nn.Sequential(
            nn.Conv1d(input_dim, hidden_dim, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_dim),
            nn.Dropout(dropout),
            nn.Conv1d(hidden_dim, hidden_dim, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_dim),
            nn.Dropout(dropout),
            nn.Conv1d(hidden_dim, hidden_dim, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_dim),
            nn.Dropout(dropout),
        )

        self.segment_layer = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

        self.embedding_layer = nn.Linear(hidden_dim, embedding_dim)

        self.attention = nn.Sequential(
            nn.Linear(hidden_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 1),
        )

    def _apply_attention(self, x: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        """Apply self-attention pooling."""
        attn_weights = self.attention(x)

        mask = torch.arange(x.shape[1]).unsqueeze(0).to(x.device) >= lengths.unsqueeze(
            1
        )
        attn_weights = attn_weights.masked_fill(mask, float("-inf"))

        attn_weights = F.softmax(attn_weights, dim=1)

        x = torch.bmm(attn_weights.transpose(1, 2), x).squeeze(1)

        return x

    def forward(
        self,
        features: torch.Tensor,
        lengths: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Extract speaker embeddings.

        Args:
            features: Input features (batch, seq_len, input_dim)
            lengths: Sequence lengths

        Returns:
            Speaker embeddings (batch, embedding_dim)
        """
        x = features.transpose(1, 2)

        x = self.frame_layer(x)

        x = x.transpose(1, 2)

        if lengths is not None:
            x = self._apply_attention(x, lengths)
        else:
            x = x.mean(dim=1)

        x = self.segment_layer(x)

        embeddings = self.embedding_layer(x)

        return F.normalize(embeddings, p=2, dim=1)


class RawNetEncoder(nn.Module):
    """RawNet-style speaker encoder.

    Processes raw waveform directly without explicit feature extraction.
    """

    def __init__(
        self,
        sample_rate: int = 16000,
        embedding_dim: int = 256,
        hidden_dim: int = 512,
        dropout: float = 0.2,
    ):
        super().__init__()

        self.sample_rate = sample_rate

        self.conv_blocks = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Conv1d(1, 16, kernel_size=11, stride=4, padding=5),
                    nn.BatchNorm1d(16),
                    nn.ReLU(),
                    nn.Dropout(dropout),
                ),
                nn.Sequential(
                    nn.Conv1d(16, 32, kernel_size=7, stride=2, padding=3),
                    nn.BatchNorm1d(32),
                    nn.ReLU(),
                    nn.Dropout(dropout),
                ),
                nn.Sequential(
                    nn.Conv1d(32, 64, kernel_size=5, stride=2, padding=2),
                    nn.BatchNorm1d(64),
                    nn.ReLU(),
                    nn.Dropout(dropout),
                ),
                nn.Sequential(
                    nn.Conv1d(64, 128, kernel_size=3, stride=2, padding=1),
                    nn.BatchNorm1d(128),
                    nn.ReLU(),
                    nn.Dropout(dropout),
                ),
            ]
        )

        self.gru = nn.GRU(
            input_size=128,
            hidden_size=hidden_dim,
            num_layers=2,
            batch_first=True,
            bidirectional=True,
            dropout=dropout,
        )

        self.embedding = nn.Linear(hidden_dim * 2, embedding_dim)

        self.attention = nn.Sequential(
            nn.Linear(hidden_dim * 2, 128),
            nn.ReLU(),
            nn.Linear(128, 1),
        )

    def _apply_attention(
        self, x: torch.Tensor, mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Apply attention pooling."""
        attn = self.attention(x)

        if mask is not None:
            attn = attn.masked_fill(mask.unsqueeze(-1), float("-inf"))

        attn = F.softmax(attn, dim=1)

        x = torch.sum(x * attn, dim=1)

        return x

    def forward(
        self,
        audio: torch.Tensor,
        length: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Extract speaker embeddings from raw audio.

        Args:
            audio: Raw audio waveform (batch, n_samples)
            length: Optional actual sequence length

        Returns:
            Speaker embeddings (batch, embedding_dim)
        """
        x = audio.unsqueeze(1)

        for block in self.conv_blocks:
            x = block(x)

        x = x.transpose(1, 2)

        if length is not None:
            seq_len = x.shape[1]
            mask = (
                torch.arange(seq_len).to(x.device) >= (length / 4).unsqueeze(1).long()
            )
            x = self._apply_attention(x, mask)
        else:
            x = x.mean(dim=1)

        gru_out, _ = self.gru(x.unsqueeze(1))

        embeddings = self.embedding(gru_out.squeeze(1))

        return F.normalize(embeddings, p=2, dim=1)


class SpeakerVerification:
    """Speaker verification module.

    Compares two audio samples to determine if they
    are from the same speaker.
    """

    def __init__(
        self,
        encoder: Optional[nn.Module] = None,
        embedding_dim: int = 256,
        threshold: float = 0.5,
    ):
        if encoder is None:
            encoder = SpeakerEncoder(
                input_dim=40,
                embedding_dim=embedding_dim,
            )

        self.encoder = encoder
        self.threshold = threshold

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.encoder.to(self.device)

    def extract_embedding(self, audio: torch.Tensor) -> torch.Tensor:
        """Extract speaker embedding from audio.

        Args:
            audio: Audio waveform (n_samples,) or (batch, n_samples)

        Returns:
            Speaker embedding
        """
        self.encoder.eval()

        if audio.dim() == 1:
            audio = audio.unsqueeze(0)

        audio = audio.to(self.device)

        with torch.no_grad():
            if audio.shape[1] > 16000 * 3:
                audio = audio[:, : 16000 * 3]

            mel_spec = self._compute_mel_features(audio)

            embedding = self.encoder(mel_spec)

        return embedding.cpu()

    def _compute_mel_features(self, audio: torch.Tensor) -> torch.Tensor:
        """Compute mel spectrogram features."""
        n_fft = 512
        hop_length = 160
        n_mels = 40

        window = torch.hann_window(n_fft).to(audio.device)

        spec = torch.stft(
            audio,
            n_fft=n_fft,
            hop_length=hop_length,
            win_length=n_fft,
            window=window,
            return_complex=True,
        )
        spec = torch.abs(spec)

        mel_basis = torch.nn.functional.interpolate(
            torch.rand(n_mels, n_fft // 2 + 1).unsqueeze(0).to(audio.device),
            size=n_fft // 2 + 1,
            mode="bilinear",
            align_corners=False,
        ).squeeze(0)

        mel_spec = torch.matmul(mel_basis, spec)
        mel_spec = torch.log(mel_spec + 1e-9)

        return mel_spec.transpose(1, 2)

    def verify(
        self,
        audio1: torch.Tensor,
        audio2: torch.Tensor,
        threshold: Optional[float] = None,
    ) -> Tuple[bool, float]:
        """Verify if two audio samples are from the same speaker.

        Args:
            audio1: First audio waveform
            audio2: Second audio waveform
            threshold: Decision threshold (uses default if None)

        Returns:
            Tuple of (is_same_speaker, similarity_score)
        """
        threshold = threshold or self.threshold

        emb1 = self.extract_embedding(audio1)
        emb2 = self.extract_embedding(audio2)

        similarity = F.cosine_similarity(emb1, emb2, dim=1).item()

        is_same = similarity >= threshold

        return is_same, similarity

    def compare_enrollments(
        self,
        audio: torch.Tensor,
        enrollment_embeddings: Dict[str, torch.Tensor],
    ) -> Tuple[Optional[str], float]:
        """Compare audio to enrolled speakers.

        Args:
            audio: Audio waveform to verify
            enrollment_embeddings: Dictionary of speaker_id -> embedding

        Returns:
            Tuple of (best_match_speaker_id, similarity)
        """
        if not enrollment_embeddings:
            return None, 0.0

        audio_embedding = self.extract_embedding(audio)

        best_speaker = None
        best_similarity = -1.0

        for speaker_id, enrolled_emb in enrollment_embeddings.items():
            similarity = F.cosine_similarity(
                audio_embedding,
                enrolled_emb.to(audio_embedding.device),
                dim=1,
            ).item()

            if similarity > best_similarity:
                best_similarity = similarity
                best_speaker = speaker_id

        return best_speaker, best_similarity


class SpeakerIdentification:
    """Speaker identification module.

    Identifies the speaker from a list of enrolled speakers.
    """

    def __init__(
        self,
        encoder: Optional[nn.Module] = None,
        embedding_dim: int = 256,
    ):
        if encoder is None:
            encoder = SpeakerEncoder(
                input_dim=40,
                embedding_dim=embedding_dim,
            )

        self.encoder = encoder
        self.enrolled_speakers: Dict[str, torch.Tensor] = {}

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.encoder.to(self.device)

    def enroll_speaker(
        self,
        speaker_id: str,
        audio_samples: List[torch.Tensor],
    ) -> torch.Tensor:
        """Enroll a speaker with multiple audio samples.

        Args:
            speaker_id: Unique speaker identifier
            audio_samples: List of audio waveforms for enrollment

        Returns:
            Enrolled speaker embedding
        """
        self.encoder.eval()

        embeddings = []

        for audio in audio_samples:
            audio = audio.to(self.device)

            with torch.no_grad():
                mel_spec = self._compute_mel_features(audio.unsqueeze(0))
                embedding = self.encoder(mel_spec)
                embeddings.append(embedding.cpu())

        embeddings = torch.stack(embeddings)

        avg_embedding = embeddings.mean(dim=0)
        avg_embedding = F.normalize(avg_embedding, p=2, dim=1)

        self.enrolled_speakers[speaker_id] = avg_embedding

        return avg_embedding

    def _compute_mel_features(self, audio: torch.Tensor) -> torch.Tensor:
        """Compute mel spectrogram features."""
        n_fft = 512
        hop_length = 160
        n_mels = 40

        window = torch.hann_window(n_fft).to(audio.device)

        spec = torch.stft(
            audio,
            n_fft=n_fft,
            hop_length=hop_length,
            win_length=n_fft,
            window=window,
            return_complex=True,
        )
        spec = torch.abs(spec)

        mel_basis = torch.nn.functional.interpolate(
            torch.rand(n_mels, n_fft // 2 + 1).unsqueeze(0).to(audio.device),
            size=n_fft // 2 + 1,
            mode="bilinear",
            align_corners=False,
        ).squeeze(0)

        mel_spec = torch.matmul(mel_basis, spec)
        mel_spec = torch.log(mel_spec + 1e-9)

        return mel_spec.transpose(1, 2)

    def identify(
        self,
        audio: torch.Tensor,
    ) -> Tuple[Optional[str], Dict[str, float]]:
        """Identify the speaker from enrolled speakers.

        Args:
            audio: Audio waveform to identify

        Returns:
            Tuple of (speaker_id, similarities_dict)
        """
        if not self.enrolled_speakers:
            return None, {}

        self.encoder.eval()

        audio = audio.to(self.device)

        with torch.no_grad():
            mel_spec = self._compute_mel_features(audio.unsqueeze(0))
            query_embedding = self.encoder(mel_spec).cpu()

        similarities = {}

        for speaker_id, enrolled_emb in self.enrolled_speakers.items():
            similarity = F.cosine_similarity(
                query_embedding,
                enrolled_emb.to(query_embedding.device),
                dim=1,
            ).item()

            similarities[speaker_id] = similarity

        best_speaker = max(similarities, key=similarities.get)

        return best_speaker, similarities

    def remove_speaker(self, speaker_id: str) -> bool:
        """Remove a speaker from enrollment.

        Args:
            speaker_id: Speaker to remove

        Returns:
            True if removed, False if not found
        """
        if speaker_id in self.enrolled_speakers:
            del self.enrolled_speakers[speaker_id]
            return True
        return False

    def get_num_enrolled(self) -> int:
        """Get number of enrolled speakers."""
        return len(self.enrolled_speakers)


class AngularMarginLoss(nn.Module):
    """Angular margin loss for speaker verification (AAM-Softmax)."""

    def __init__(
        self,
        embedding_dim: int = 256,
        num_classes: int = 1000,
        margin: float = 0.3,
        scale: float = 30.0,
    ):
        super().__init__()

        self.embedding_dim = embedding_dim
        self.num_classes = num_classes
        self.margin = margin
        self.scale = scale

        self.weight = nn.Parameter(torch.FloatTensor(num_classes, embedding_dim))
        nn.init.xavier_uniform_(self.weight)

    def forward(
        self,
        embeddings: torch.Tensor,
        labels: torch.Tensor,
    ) -> torch.Tensor:
        """Compute angular margin loss.

        Args:
            embeddings: Input embeddings (batch, embedding_dim)
            labels: Speaker labels (batch,)

        Returns:
            Loss value
        """
        normalized_weights = F.normalize(self.weight, p=2, dim=1)

        cosine = F.linear(embeddings, normalized_weights)

        theta = torch.acos(torch.clamp(cosine, -1.0 + 1e-7, 1.0 - 1e-7))

        target_cosine = torch.cos(theta + self.margin)

        one_hot = F.one_hot(labels, num_classes=self.num_classes).float()

        logits = (one_hot * target_cosine) + ((1.0 - one_hot) * cosine)

        logits = logits * self.scale

        loss = F.cross_entropy(logits, labels)

        return loss


def create_speaker_encoder(
    encoder_type: str = "tdnn",
    embedding_dim: int = 256,
    **kwargs,
) -> nn.Module:
    """Factory function to create speaker encoder.

    Args:
        encoder_type: Type of encoder ("tdnn" or "rawnet")
        embedding_dim: Embedding dimension
        **kwargs: Additional arguments

    Returns:
        Speaker encoder model
    """
    if encoder_type == "tdnn":
        return SpeakerEncoder(
            input_dim=kwargs.get("input_dim", 40),
            embedding_dim=embedding_dim,
            hidden_dim=kwargs.get("hidden_dim", 512),
            num_layers=kwargs.get("num_layers", 3),
            dropout=kwargs.get("dropout", 0.2),
        )
    elif encoder_type == "rawnet":
        return RawNetEncoder(
            sample_rate=kwargs.get("sample_rate", 16000),
            embedding_dim=embedding_dim,
            hidden_dim=kwargs.get("hidden_dim", 512),
            dropout=kwargs.get("dropout", 0.2),
        )
    else:
        raise ValueError(f"Unknown encoder type: {encoder_type}")
