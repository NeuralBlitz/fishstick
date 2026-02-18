"""
Voice Conversion Data Types Module.

Comprehensive type definitions for voice conversion and speech synthesis.
Provides type hints, dataclasses, and protocols for audio processing,
speaker embeddings, and voice conversion models.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum, auto
from pathlib import Path
from typing import (
    Any,
    Dict,
    Generic,
    List,
    NamedTuple,
    Optional,
    Protocol,
    Tuple,
    TypeVar,
    Union,
)

import numpy as np

T = TypeVar("T")


class AudioFormat(Enum):
    """Supported audio formats."""

    WAV = auto()
    PCM = auto()
    FLAC = auto()
    MP3 = auto()
    OGG = auto()


class ConversionMode(Enum):
    """Voice conversion modes."""

    ANY_TO_ONE = auto()
    ONE_TO_ONE = auto()
    ANY_TO_ANY = auto()
    STYLE_TRANSFER = auto()
    PROSODY_ONLY = auto()


class VocoderType(Enum):
    """Neural vocoder types."""

    WAVENET = auto()
    WAVEGLOW = auto()
    PARALLEL_WAVEGAN = auto()
    HIFIGAN = auto()
    GRIFFIN_LIM = auto()


class SpeakerEmbeddingType(Enum):
    """Speaker embedding types."""

    XVECTOR = auto()
    DVECTOR = auto()
    XVECTOR_ECAPA = auto()


class PitchExtractionMethod(Enum):
    """Pitch extraction methods."""

    DIO = auto()
    Harvest = auto()
    PYIN = auto()
    CREPE = auto()
    SWIP = auto()


@dataclass
class AudioConfig:
    """Audio configuration for voice conversion."""

    sample_rate: int = 22050
    hop_length: int = 256
    win_length: int = 1024
    n_fft: int = 1024
    n_mels: int = 80
    fmin: float = 0.0
    fmax: float = 8000.0
    center: bool = True
    dtype: str = "float32"

    def __post_init__(self) -> None:
        if self.sample_rate <= 0:
            raise ValueError("sample_rate must be positive")
        if self.hop_length <= 0:
            raise ValueError("hop_length must be positive")


@dataclass
class SpectralFeatures:
    """Container for spectral features."""

    melspectrogram: Optional[np.ndarray] = None
    mfcc: Optional[np.ndarray] = None
    spectrogram: Optional[np.ndarray] = None
    magnitudes: Optional[np.ndarray] = None
    phase: Optional[np.ndarray] = None

    frame_rate: Optional[int] = None
    n_fft: int = 1024
    hop_length: int = 256
    n_mels: int = 80
    n_mfcc: int = 40

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        result: Dict[str, Any] = {
            "n_fft": self.n_fft,
            "hop_length": self.hop_length,
            "n_mels": self.n_mels,
            "n_mfcc": self.n_mfcc,
        }
        if self.melspectrogram is not None:
            result["melspectrogram_shape"] = self.melspectrogram.shape
        if self.mfcc is not None:
            result["mfcc_shape"] = self.mfcc.shape
        if self.spectrogram is not None:
            result["spectrogram_shape"] = self.spectrogram.shape
        return result


@dataclass
class ProsodyFeatures:
    """Container for prosodic features."""

    pitch: Optional[np.ndarray] = None
    duration: Optional[np.ndarray] = None
    energy: Optional[np.ndarray] = None
    voicing: Optional[np.ndarray] = None

    pitch_mean: Optional[float] = None
    pitch_std: Optional[float] = None
    energy_mean: Optional[float] = None
    energy_std: Optional[float] = None

    frame_rate: Optional[int] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        result: Dict[str, Any] = {}
        if self.pitch is not None:
            result["pitch_shape"] = self.pitch.shape
            result["pitch_mean"] = self.pitch_mean
            result["pitch_std"] = self.pitch_std
        if self.energy is not None:
            result["energy_shape"] = self.energy.shape
            result["energy_mean"] = self.energy_mean
            result["energy_std"] = self.energy_std
        return result


@dataclass
class SpeakerEmbedding:
    """Container for speaker embedding."""

    embedding: np.ndarray
    speaker_id: Optional[str] = None
    embedding_type: SpeakerEmbeddingType = SpeakerEmbeddingType.XVECTOR

    @property
    def shape(self) -> Tuple[int, ...]:
        """Get embedding shape."""
        return self.embedding.shape

    @property
    def dimension(self) -> int:
        """Get embedding dimension."""
        return int(np.prod(self.embedding.shape))

    def similarity(self, other: SpeakerEmbedding) -> float:
        """Compute cosine similarity with another embedding."""
        a = self.embedding.flatten()
        b = other.embedding.flatten()
        return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-8))

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "speaker_id": self.speaker_id,
            "embedding_type": self.embedding_type.name,
            "dimension": self.dimension,
            "shape": list(self.shape),
        }


@dataclass
class VoiceConversionResult:
    """Result of voice conversion."""

    converted_audio: np.ndarray
    source_speaker: Optional[str] = None
    target_speaker: Optional[str] = None
    source_embedding: Optional[SpeakerEmbedding] = None
    target_embedding: Optional[SpeakerEmbedding] = None

    spectral_features: Optional[SpectralFeatures] = None
    prosody_features: Optional[ProsodyFeatures] = None

    conversion_time_ms: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def audio_length_sec(self) -> float:
        """Get audio length in seconds."""
        return len(self.converted_audio) / 22050.0


@dataclass
class AudioSample:
    """Single audio sample with metadata."""

    waveform: np.ndarray
    sample_rate: int = 22050
    speaker_id: Optional[str] = None
    text: Optional[str] = None
    emotion: Optional[str] = None

    @property
    def duration_sec(self) -> float:
        """Get duration in seconds."""
        return len(self.waveform) / self.sample_rate

    @property
    def num_samples(self) -> int:
        """Get number of samples."""
        return len(self.waveform)

    def normalize(self, target_level_db: float = -20.0) -> AudioSample:
        """Normalize audio to target level."""
        rms = np.sqrt(np.mean(self.waveform**2))
        if rms > 0:
            target_rms = 10 ** (target_level_db / 20.0)
            scale = target_rms / rms
            waveform = self.waveform * scale
        else:
            waveform = self.waveform
        return AudioSample(
            waveform=waveform,
            sample_rate=self.sample_rate,
            speaker_id=self.speaker_id,
            text=self.text,
            emotion=self.emotion,
        )


@dataclass
class BatchAudio:
    """Batch of audio samples."""

    waveforms: List[np.ndarray]
    sample_rate: int = 22050
    speaker_ids: Optional[List[Optional[str]]] = None
    texts: Optional[List[Optional[str]]] = None
    metadata: List[Dict[str, Any]] = field(default_factory=list)

    def __len__(self) -> int:
        """Get batch size."""
        return len(self.waveforms)

    @property
    def max_length(self) -> int:
        """Get maximum waveform length in batch."""
        return max(len(w) for w in self.waveforms)

    @property
    def min_length(self) -> int:
        """Get minimum waveform length in batch."""
        return min(len(w) for w in self.waveforms)


class AudioLoader(Protocol):
    """Protocol for audio loaders."""

    def load(self, path: Union[str, Path]) -> AudioSample:
        """Load audio from path."""
        ...


class Vocoder(ABC):
    """Abstract base class for neural vocoders."""

    @abstractmethod
    def synthesize(self, mel: np.ndarray) -> np.ndarray:
        """Synthesize waveform from mel spectrogram."""
        pass

    @abstractmethod
    def inference(self, mel: np.ndarray) -> np.ndarray:
        """Run inference."""
        pass


class SpeakerEncoder(ABC):
    """Abstract base class for speaker encoders."""

    @abstractmethod
    def encode(self, audio: np.ndarray, sample_rate: int) -> SpeakerEmbedding:
        """Encode audio to speaker embedding."""
        pass

    @abstractmethod
    def compute_similarity(
        self, embedding1: SpeakerEmbedding, embedding2: SpeakerEmbedding
    ) -> float:
        """Compute similarity between embeddings."""
        pass


class VoiceConverter(ABC):
    """Abstract base class for voice converters."""

    @abstractmethod
    def convert(
        self,
        source_audio: np.ndarray,
        target_speaker: Union[str, SpeakerEmbedding],
    ) -> VoiceConversionResult:
        """Convert voice from source to target speaker."""
        pass


class PitchExtractor(ABC):
    """Abstract base class for pitch extractors."""

    @abstractmethod
    def extract(self, audio: np.ndarray, sample_rate: int) -> np.ndarray:
        """Extract pitch contour from audio."""
        pass

    @abstractmethod
    def extract_f0(
        self, audio: np.ndarray, sample_rate: int, frame_length: int
    ) -> np.ndarray:
        """Extract F0 contour."""
        pass


class SpectralConverter(ABC):
    """Abstract base class for spectral converters."""

    @abstractmethod
    def convert_spectral(
        self,
        source_mel: np.ndarray,
        source_embedding: SpeakerEmbedding,
        target_embedding: SpeakerEmbedding,
    ) -> np.ndarray:
        """Convert spectral features."""
        pass


class StyleEncoder(ABC):
    """Abstract base class for style encoders."""

    @abstractmethod
    def encode_style(
        self, audio: np.ndarray, sample_rate: int
    ) -> Dict[str, np.ndarray]:
        """Encode style features from audio."""
        pass


class ProsodyConverter(ABC):
    """Abstract base class for prosody converters."""

    @abstractmethod
    def convert_prosody(
        self,
        source_prosody: ProsodyFeatures,
        target_prosody: Optional[ProsodyFeatures] = None,
    ) -> ProsodyFeatures:
        """Convert prosody features."""
        pass


class AudioProcessor:
    """Audio processing utilities."""

    @staticmethod
    def resample(audio: np.ndarray, orig_sr: int, target_sr: int) -> np.ndarray:
        """Resample audio to target sample rate."""
        if orig_sr == target_sr:
            return audio

        duration = len(audio) / orig_sr
        num_samples = int(duration * target_sr)

        indices = np.linspace(0, len(audio) - 1, num_samples)
        resampled = np.interp(indices, np.arange(len(audio)), audio)
        return resampled

    @staticmethod
    def compute_loudness(audio: np.ndarray) -> float:
        """Compute loudness in LUFS (simplified)."""
        return float(-0.691 + 10 * np.log10(np.mean(audio**2) + 1e-10))

    @staticmethod
    def apply_fade(
        audio: np.ndarray,
        fade_in_samples: int = 0,
        fade_out_samples: int = 0,
    ) -> np.ndarray:
        """Apply fade in/out to audio."""
        result = audio.copy()
        if fade_in_samples > 0:
            fade_in = np.linspace(0, 1, fade_in_samples)
            result[:fade_in_samples] *= fade_in
        if fade_out_samples > 0:
            fade_out = np.linspace(1, 0, fade_out_samples)
            result[-fade_out_samples:] *= fade_out
        return result


class FeatureExtractor:
    """Feature extraction utilities."""

    @staticmethod
    def extract_mfcc(
        audio: np.ndarray,
        sample_rate: int = 22050,
        n_mfcc: int = 40,
        n_fft: int = 1024,
        hop_length: int = 256,
    ) -> np.ndarray:
        """Extract MFCC features."""
        try:
            import librosa

            mfcc = librosa.feature.mfcc(
                y=audio,
                sr=sample_rate,
                n_mfcc=n_mfcc,
                n_fft=n_fft,
                hop_length=hop_length,
            )
            return mfcc
        except ImportError:
            return np.zeros((n_mfcc, len(audio) // hop_length))

    @staticmethod
    def extract_mel_spectrogram(
        audio: np.ndarray,
        sample_rate: int = 22050,
        n_fft: int = 1024,
        hop_length: int = 256,
        n_mels: int = 80,
    ) -> np.ndarray:
        """Extract mel spectrogram."""
        try:
            import librosa

            mel = librosa.feature.melspectrogram(
                y=audio,
                sr=sample_rate,
                n_fft=n_fft,
                hop_length=hop_length,
                n_mels=n_mels,
            )
            return librosa.power_to_db(mel, ref=np.max)
        except ImportError:
            return np.zeros((n_mels, len(audio) // hop_length))

    @staticmethod
    def extract_pitch(
        audio: np.ndarray,
        sample_rate: int = 22050,
        hop_length: int = 256,
    ) -> np.ndarray:
        """Extract pitch contour using DIO."""
        try:
            import librosa
            import pyworld

            f0, _ = pyworld.dio(
                audio.astype(np.float64),
                sample_rate,
                frame_period=hop_length / sample_rate * 1000,
            )
            return f0
        except ImportError:
            try:
                import librosa

                f0 = librosa.yin(
                    audio,
                    fmin=librosa.note_to_hz("C2"),
                    fmax=librosa.note_to_hz("C7"),
                    hop_length=hop_length,
                )
                return f0
            except ImportError:
                return np.zeros(len(audio) // hop_length)


__all__ = [
    "AudioFormat",
    "ConversionMode",
    "VocoderType",
    "SpeakerEmbeddingType",
    "PitchExtractionMethod",
    "AudioConfig",
    "SpectralFeatures",
    "ProsodyFeatures",
    "SpeakerEmbedding",
    "VoiceConversionResult",
    "AudioSample",
    "BatchAudio",
    "AudioLoader",
    "Vocoder",
    "SpeakerEncoder",
    "VoiceConverter",
    "PitchExtractor",
    "SpectralConverter",
    "StyleEncoder",
    "ProsodyConverter",
    "AudioProcessor",
    "FeatureExtractor",
]
