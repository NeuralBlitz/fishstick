"""
Voice Conversion Module for Fishstick AI Framework.

This module provides comprehensive voice conversion and speech synthesis tools:
- Spectral Mapping: Transform spectral features between speakers
- Neural Vocoders: Convert mel spectrograms to waveforms
- Voice Style Transfer: Transfer voice style characteristics
- Speaker Encoding: Extract speaker embeddings
- Prosody Conversion: Convert pitch, duration, and energy

Main Classes:
    VoiceConverter: One-to-one voice conversion
    AnyToAnyConverter: Many-to-many voice conversion
    VoiceConversionPipeline: End-to-end pipeline

Example:
    >>> from fishstick.voice_conversion import VoiceConversionPipeline, VoiceConversionConfig
    >>>
    >>> config = VoiceConversionConfig(
    ...     n_mels=80,
    ...     num_speakers=10,
    ...     use_vocoder=True,
    ... )
    >>> pipeline = VoiceConversionPipeline(config)
    >>>
    >>> # Convert voice
    >>> converted = pipeline(source_mel, target_speaker_id=1)
"""

from fishstick.voice_conversion.spectral_mapping import (
    SpectralConfig,
    SpectralNormalization,
    SpectralMappingNetwork,
    FrequencyMasking,
    MelSpectralMapper,
    ConditionalSpectralMapper,
)

from fishstick.voice_conversion.neural_vocoders import (
    VocoderConfig,
    WaveNetVocoder,
    ParallelWaveGANGenerator,
    HiFiGANVocoder,
    MultiPeriodDiscriminator,
    MultiScaleDiscriminator,
    create_vocoder,
)

from fishstick.voice_conversion.voice_style_transfer import (
    StyleTransferConfig,
    StyleAdaptationLayer,
    StyleEncoder,
    ReferenceEncoder,
    StyleTransferNetwork,
    ExpressiveVoiceConverter,
    ProsodyEncoder,
    DurationPredictor,
    create_style_transfer_network,
)

from fishstick.voice_conversion.speaker_encoding import (
    SpeakerConfig,
    SpeakerEmbeddingNetwork,
    SpeakerEncoderAdvanced,
    GE2ELoss,
    AngularProtoLoss,
    AAMSoftmax,
    SpeakerEncoderWithLoss,
    create_speaker_encoder,
)

from fishstick.voice_conversion.prosody_conversion import (
    ProsodyConfig,
    PitchExtractor,
    PitchConverter,
    DurationConverter,
    EnergyConverter,
    ProsodyExtractor,
    ProsodyConverter,
    HierarchicalProsodyConverter,
    ContourPredictor,
    create_prosody_converter,
)

from fishstick.voice_conversion.conversion_pipeline import (
    VoiceConversionConfig,
    VoiceConverter,
    AnyToAnyConverter,
    SpeakerVerification,
    VoiceConversionPipeline,
    ParallelVoiceConverter,
    CycleConsistencyLoss,
    SpeakerClassificationLoss,
    create_voice_converter,
)

from fishstick.voice_conversion.types import (
    AudioFormat,
    ConversionMode,
    VocoderType,
    SpeakerEmbeddingType,
    PitchExtractionMethod,
    AudioConfig,
    SpectralFeatures,
    ProsodyFeatures,
    SpeakerEmbedding,
    VoiceConversionResult,
    AudioSample,
    BatchAudio,
)

__all__ = [
    # Spectral Mapping
    "SpectralConfig",
    "SpectralNormalization",
    "SpectralMappingNetwork",
    "FrequencyMasking",
    "MelSpectralMapper",
    "ConditionalSpectralMapper",
    # Neural Vocoders
    "VocoderConfig",
    "WaveNetVocoder",
    "ParallelWaveGANGenerator",
    "HiFiGANVocoder",
    "MultiPeriodDiscriminator",
    "MultiScaleDiscriminator",
    "create_vocoder",
    # Voice Style Transfer
    "StyleTransferConfig",
    "StyleAdaptationLayer",
    "StyleEncoder",
    "ReferenceEncoder",
    "StyleTransferNetwork",
    "ExpressiveVoiceConverter",
    "ProsodyEncoder",
    "DurationPredictor",
    "create_style_transfer_network",
    # Speaker Encoding
    "SpeakerConfig",
    "SpeakerEmbeddingNetwork",
    "SpeakerEncoderAdvanced",
    "GE2ELoss",
    "AngularProtoLoss",
    "AAMSoftmax",
    "SpeakerEncoderWithLoss",
    "create_speaker_encoder",
    # Prosody Conversion
    "ProsodyConfig",
    "PitchExtractor",
    "PitchConverter",
    "DurationConverter",
    "EnergyConverter",
    "ProsodyExtractor",
    "ProsodyConverter",
    "HierarchicalProsodyConverter",
    "ContourPredictor",
    "create_prosody_converter",
    # Pipeline
    "VoiceConversionConfig",
    "VoiceConverter",
    "AnyToAnyConverter",
    "SpeakerVerification",
    "VoiceConversionPipeline",
    "ParallelVoiceConverter",
    "CycleConsistencyLoss",
    "SpeakerClassificationLoss",
    "create_voice_converter",
    # Types
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
]
