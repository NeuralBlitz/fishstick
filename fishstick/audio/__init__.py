"""
fishstick Audio Module

Audio processing, feature extraction, and audio deep learning models.
"""

from fishstick.audio.models import (
    AudioClassifier,
    WaveNetAudio,
    TransformerAudio,
    VQVAE,
    AudioAutoencoder,
)
from fishstick.audio.features import (
    MelSpectrogram,
    MFCC,
    Spectrogram,
    ChromaFeatures,
    SpectralContrast,
)
from fishstick.audio.preprocessing import (
    AudioLoader,
    AudioNormalizer,
    TimeStretch,
    PitchShift,
    AddNoise,
)

__all__ = [
    # Models
    "AudioClassifier",
    "WaveNetAudio",
    "TransformerAudio",
    "VQVAE",
    "AudioAutoencoder",
    # Features
    "MelSpectrogram",
    "MFCC",
    "Spectrogram",
    "ChromaFeatures",
    "SpectralContrast",
    # Preprocessing
    "AudioLoader",
    "AudioNormalizer",
    "TimeStretch",
    "PitchShift",
    "AddNoise",
]
