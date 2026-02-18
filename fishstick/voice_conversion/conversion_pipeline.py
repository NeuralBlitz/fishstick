"""
Voice Conversion Pipeline Module

This module provides complete voice conversion pipelines:
- VoiceConverter: Main voice conversion class
- AnyToAnyConverter: Many-to-many voice conversion
- VoiceConversionPipeline: End-to-end pipeline
- SpeakerVerification: Quality assessment
"""

from typing import Optional, Tuple, Dict, Any, List
from dataclasses import dataclass, field
from abc import ABC, abstractmethod

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from fishstick.voice_conversion.spectral_mapping import (
    SpectralMappingNetwork,
    SpectralConfig,
    MelSpectralMapper,
)
from fishstick.voice_conversion.neural_vocoders import (
    HiFiGANVocoder,
    VocoderConfig,
)
from fishstick.voice_conversion.speaker_encoding import (
    SpeakerEmbeddingNetwork,
    SpeakerConfig,
)
from fishstick.voice_conversion.prosody_conversion import (
    ProsodyConverter,
    ProsodyConfig,
)
from fishstick.voice_conversion.voice_style_transfer import (
    ExpressiveVoiceConverter,
    StyleTransferConfig,
)


@dataclass
class VoiceConversionConfig:
    """Configuration for complete voice conversion pipeline."""

    n_mels: int = 80
    hidden_dim: int = 256
    embedding_dim: int = 256
    sample_rate: int = 22050
    hop_length: int = 256

    use_spectral_mapping: bool = True
    use_style_transfer: bool = True
    use_prosody: bool = True
    use_vocoder: bool = True

    vocoder_type: str = "hifigan"
    encoder_type: str = "tdnn"

    speaker_embedding_dim: int = 256
    num_speakers: int = 0

    prosody_method: str = "full"


class VoiceConverter(nn.Module):
    """Main voice converter for one-to-one conversion.

    Converts voice from source speaker to target speaker using
    spectral mapping and neural vocoder.

    Args:
        config: VoiceConversionConfig with pipeline parameters
    """

    def __init__(self, config: VoiceConversionConfig):
        super().__init__()
        self.config = config

        if config.use_spectral_mapping:
            self.spectral_mapper = MelSpectralMapper(
                num_mels=config.n_mels,
                hidden_channels=config.hidden_dim,
            )

        if config.use_style_transfer:
            style_config = StyleTransferConfig(
                n_mels=config.n_mels,
                hidden_dim=config.hidden_dim,
                style_dim=config.embedding_dim,
                speaker_embedding_dim=config.speaker_embedding_dim,
            )
            self.style_transfer = ExpressiveVoiceConverter(style_config)

        if config.use_prosody:
            prosody_config = ProsodyConfig(
                n_mels=config.n_mels,
                sample_rate=config.sample_rate,
                hop_length=config.hop_length,
            )
            self.prosody_converter = ProsodyConverter(prosody_config)

        if config.use_vocoder:
            vocoder_config = VocoderConfig(
                n_mels=config.n_mels,
                sample_rate=config.sample_rate,
                hop_length=config.hop_length,
            )
            self.vocoder = HiFiGANVocoder(vocoder_config)

    def forward(
        self,
        source_mel: Tensor,
        target_speaker_embedding: Optional[Tensor] = None,
        ref_mel: Optional[Tensor] = None,
    ) -> Tensor:
        converted = source_mel

        if hasattr(self, "spectral_mapper"):
            converted = self.spectral_mapper(converted)

        if hasattr(self, "style_transfer") and target_speaker_embedding is not None:
            converted = self.style_transfer(
                converted,
                ref_mel=ref_mel,
                speaker_embedding=target_speaker_embedding,
            )

        if hasattr(self, "vocoder"):
            converted = self.vocoder(converted)

        return converted

    def inference(
        self,
        source_mel: Tensor,
        target_speaker_embedding: Optional[Tensor] = None,
    ) -> Tensor:
        return self.forward(source_mel, target_speaker_embedding)


class AnyToAnyConverter(nn.Module):
    """Many-to-many voice converter supporting arbitrary speaker pairs.

    Uses speaker embeddings to enable conversion between any pair of speakers.

    Args:
        config: VoiceConversionConfig with pipeline parameters
    """

    def __init__(self, config: VoiceConversionConfig):
        super().__init__()
        self.config = config

        if config.num_speakers > 0:
            self.speaker_encoder = SpeakerEmbeddingNetwork(
                SpeakerConfig(
                    input_dim=config.n_mels,
                    hidden_dim=config.hidden_dim,
                    embedding_dim=config.speaker_embedding_dim,
                )
            )

        self.voice_converter = VoiceConverter(config)

        self.register_buffer(
            "speaker_embeddings",
            torch.zeros(config.num_speakers, config.speaker_embedding_dim),
        )

    def register_speaker(
        self,
        speaker_id: int,
        embedding: Tensor,
    ) -> None:
        self.speaker_embeddings[speaker_id] = embedding

    def encode_speaker(
        self,
        mel: Tensor,
    ) -> Tensor:
        if hasattr(self, "speaker_encoder"):
            return self.speaker_encoder(mel)
        return torch.zeros(
            mel.size(0), self.config.speaker_embedding_dim, device=mel.device
        )

    def forward(
        self,
        source_mel: Tensor,
        target_speaker_id: Optional[int] = None,
        target_embedding: Optional[Tensor] = None,
    ) -> Tensor:
        if target_embedding is None and target_speaker_id is not None:
            target_embedding = self.speaker_embeddings[target_speaker_id]
            target_embedding = target_embedding.unsqueeze(0).expand(
                source_mel.size(0), -1
            )

        converted = self.voice_converter(
            source_mel,
            target_speaker_embedding=target_embedding,
        )

        return converted


class SpeakerVerification(nn.Module):
    """Speaker verification for voice conversion quality assessment.

    Evaluates similarity between converted and target speaker voices.
    """

    def __init__(
        self,
        embedding_dim: int = 256,
        hidden_dim: int = 256,
    ):
        super().__init__()

        self.encoder = nn.Sequential(
            nn.Linear(embedding_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1),
        )

    def compute_similarity(
        self,
        embedding1: Tensor,
        embedding2: Tensor,
    ) -> Tensor:
        combined = torch.cat([embedding1, embedding2], dim=-1)
        score = self.encoder(combined)
        return torch.sigmoid(score)

    def forward(
        self,
        embedding1: Tensor,
        embedding2: Tensor,
    ) -> Tensor:
        return self.compute_similarity(embedding1, embedding2)


class VoiceConversionPipeline(nn.Module):
    """Complete end-to-end voice conversion pipeline.

    Integrates all components for full voice conversion from raw audio
    to converted audio.

    Args:
        config: VoiceConversionConfig with pipeline parameters
    """

    def __init__(self, config: VoiceConversionConfig):
        super().__init__()
        self.config = config

        self.converter = AnyToAnyConverter(config)

        self.verification = SpeakerVerification(
            embedding_dim=config.embedding_dim,
        )

    def extract_features(
        self,
        mel: Tensor,
    ) -> Dict[str, Tensor]:
        return {
            "mel": mel,
        }

    def convert(
        self,
        source_mel: Tensor,
        target_speaker_id: Optional[int] = None,
        target_embedding: Optional[Tensor] = None,
    ) -> Dict[str, Any]:
        converted_wav = self.converter(
            source_mel,
            target_speaker_id=target_speaker_id,
            target_embedding=target_embedding,
        )

        if target_embedding is not None:
            source_embedding = self.converter.encode_speaker(source_mel)
            similarity = self.verification(source_embedding, target_embedding)
        else:
            similarity = None

        return {
            "waveform": converted_wav,
            "similarity_score": similarity,
        }

    def forward(
        self,
        source_mel: Tensor,
        target_speaker_id: Optional[int] = None,
        target_embedding: Optional[Tensor] = None,
    ) -> Tensor:
        result = self.convert(source_mel, target_speaker_id, target_embedding)
        return result["waveform"]


class ParallelVoiceConverter(nn.Module):
    """Parallel voice converter for multi-speaker training.

    Supports parallel conversion of multiple source speakers
    to multiple target speakers.
    """

    def __init__(self, config: VoiceConversionConfig):
        super().__init__()
        self.config = config

        self.converter = VoiceConverter(config)

    def forward(
        self,
        source_mels: Tensor,
        target_embeddings: Tensor,
    ) -> Tensor:
        batch_size = source_mels.size(0)
        num_targets = target_embeddings.size(0)

        source_mels_expanded = source_mels.unsqueeze(1).expand(-1, num_targets, -1, -1)
        target_emb_expanded = target_embeddings.unsqueeze(0).expand(batch_size, -1, -1)

        converted = []
        for i in range(batch_size):
            for j in range(num_targets):
                converted.append(
                    self.converter(
                        source_mels_expanded[i, j], target_emb_expanded[i, j]
                    )
                )

        return torch.stack(converted).view(batch_size, num_targets, -1)


class CycleConsistencyLoss(nn.Module):
    """Cycle consistency loss for voice conversion.

    Ensures that converting from A->B->A recovers the original.
    """

    def __init__(self):
        super().__init__()

    def forward(
        self,
        original: Tensor,
        converted: Tensor,
        reconstructed: Tensor,
    ) -> Tensor:
        return F.l1_loss(reconstructed, original)


class SpeakerClassificationLoss(nn.Module):
    """Speaker classification loss for training."""

    def __init__(self, num_speakers: int):
        super().__init__()
        self.classifier = nn.Linear(256, num_speakers)

    def forward(
        self,
        embeddings: Tensor,
        labels: Tensor,
    ) -> Tensor:
        logits = self.classifier(embeddings)
        return F.cross_entropy(logits, labels)


def create_voice_converter(
    config: Optional[VoiceConversionConfig] = None,
    conversion_type: str = "one_to_one",
) -> nn.Module:
    """Factory function to create voice converters.

    Args:
        config: VoiceConversionConfig with parameters
        conversion_type: Type of converter ('one_to_one', 'any_to_any', 'parallel')

    Returns:
        Initialized voice converter
    """
    if config is None:
        config = VoiceConversionConfig()

    if conversion_type == "one_to_one":
        return VoiceConverter(config)
    elif conversion_type == "any_to_any":
        return AnyToAnyConverter(config)
    elif conversion_type == "parallel":
        return ParallelVoiceConverter(config)
    elif conversion_type == "pipeline":
        return VoiceConversionPipeline(config)
    else:
        raise ValueError(f"Unknown conversion type: {conversion_type}")
