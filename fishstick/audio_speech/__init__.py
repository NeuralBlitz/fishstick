"""
fishstick Audio & Speech Processing Module
==========================================

Comprehensive audio and speech processing tools:

Spectral Features:
- SpectralFeatures: General spectral analysis (centroid, rolloff, flux, flatness)
- ChromaFeatures: Chroma (pitch class) features
- SpectralContrast: Spectral contrast between peaks and valleys
- TonnetzFeatures: Tonal centroid features

Speech Features:
- SpeechFeatureExtractor: Complete speech feature extraction pipeline
- PitchExtractor: Pitch (F0) extraction (autocorrelation, YIN)
- FormantExtractor: Formant (F1-F5) extraction via LPC
- EnergyExtractor: Energy-based features
- RASTAFilter: RASTA filtering for speech features

Voice Activity Detection:
- EnergyVAD: Energy-based voice activity detection
- SpectralEntropyVAD: Spectral entropy-based VAD
- NeuralVAD: Neural network-based VAD
- HybridVAD: Hybrid VAD combining multiple methods
- VADPostProcessor: Post-processing for VAD output

Speaker Recognition:
- SpeakerEncoder: TDNN-based speaker embedding extractor
- RawNetEncoder: RawNet-style speaker encoder
- SpeakerVerification: Speaker verification module
- SpeakerIdentification: Speaker identification module
- AngularMarginLoss: AAM-Softmax loss for speaker recognition

Audio Synthesis:
- GriffinLimVocoder: Griffin-Lim vocoder for spectrogram inversion
- MelGANVocoder: MelGAN neural vocoder
- WaveNetVocoder: WaveNet neural vocoder
- NeuralVocoder: Generic neural vocoder interface
- SpeechEnhancement: Speech enhancement using U-Net
- TTSInterface: Text-to-speech abstract interface
- VocoderWrapper: Wrapper for neural vocoders
"""

from fishstick.audio_speech.spectral_features import (
    SpectralFeatures,
    SpectralConfig,
    ChromaFeatures,
    SpectralContrast,
    TonnetzFeatures,
)

from fishstick.audio_speech.speech_features import (
    SpeechFeatureExtractor,
    SpeechFeatureConfig,
    PitchExtractor,
    FormantExtractor,
    EnergyExtractor,
    RASTAFilter,
)

from fishstick.audio_speech.voice_activity import (
    EnergyVAD,
    SpectralEntropyVAD,
    NeuralVAD,
    NeuralVADWrapper,
    HybridVAD,
    VADPostProcessor,
    VADConfig,
    VADMethod,
    create_vad,
)

from fishstick.audio_speech.speaker_recognition import (
    SpeakerEncoder,
    RawNetEncoder,
    SpeakerVerification,
    SpeakerIdentification,
    AngularMarginLoss,
    SpeakerConfig,
    create_speaker_encoder,
)

from fishstick.audio_speech.audio_synthesis import (
    GriffinLimVocoder,
    MelGANVocoder,
    WaveNetVocoder,
    NeuralVocoder,
    SpeechEnhancement,
    TTSInterface,
    VocoderWrapper,
    SynthesisConfig,
    create_vocoder,
)

__all__ = [
    # Spectral Features
    "SpectralFeatures",
    "SpectralConfig",
    "ChromaFeatures",
    "SpectralContrast",
    "TonnetzFeatures",
    # Speech Features
    "SpeechFeatureExtractor",
    "SpeechFeatureConfig",
    "PitchExtractor",
    "FormantExtractor",
    "EnergyExtractor",
    "RASTAFilter",
    # Voice Activity Detection
    "EnergyVAD",
    "SpectralEntropyVAD",
    "NeuralVAD",
    "NeuralVADWrapper",
    "HybridVAD",
    "VADPostProcessor",
    "VADConfig",
    "VADMethod",
    "create_vad",
    # Speaker Recognition
    "SpeakerEncoder",
    "RawNetEncoder",
    "SpeakerVerification",
    "SpeakerIdentification",
    "AngularMarginLoss",
    "SpeakerConfig",
    "create_speaker_encoder",
    # Audio Synthesis
    "GriffinLimVocoder",
    "MelGANVocoder",
    "WaveNetVocoder",
    "NeuralVocoder",
    "SpeechEnhancement",
    "TTSInterface",
    "VocoderWrapper",
    "SynthesisConfig",
    "create_vocoder",
]
