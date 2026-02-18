"""
Audio Source Separation Module

Comprehensive audio source separation tools including time-frequency masking,
deep clustering, waveform-based separation, speaker separation, and music
source separation.

Submodules:
- base: Base classes and utilities
- time_frequency_masking: T-F masking strategies
- deep_clustering: Deep clustering methods
- waveform_separation: Waveform-based separation
- speaker_separation: Speaker diarization and separation
- music_separation: Music stem separation
- losses: Loss functions for separation
- augmentation: Data augmentation for separation
- metrics: Evaluation metrics
- training: Training utilities
- preprocessing: Audio preprocessing
- postprocessing: Audio postprocessing
- blocks: Model building blocks
"""

from fishstick.audio_separation.base import (
    SeparationResult,
    SeparationModel,
    Separator,
    STFT,
    SeparationMetrics,
    AudioMixer,
)

from fishstick.audio_separation.time_frequency_masking import (
    TimeFrequencyMask,
    IdealBinaryMask,
    IdealRatioMask,
    PhaseSensitiveMask,
    ComplexMask,
    WienerFilter,
    TFMaskingNetwork,
    IBMEstimator,
)

from fishstick.audio_separation.deep_clustering import (
    DeepClustering,
    EmbeddingExtractor,
    ClusterAssigner,
    DiscriminativeLoss,
)

from fishstick.audio_separation.waveform_separation import (
    WaveformSeparation,
    ConvTasNet,
    DualPathRNN,
    TemporalConvNet,
    SkipConnection,
)

from fishstick.audio_separation.speaker_separation import (
    SpeakerSeparator,
    SpeakerBeam,
    SpeakerRecognition,
    VoiceActivityDetector,
)

from fishstick.audio_separation.music_separation import (
    MusicSeparator,
    Demucs,
    OpenUnmix,
    XUMX,
    BandSplit,
)

try:
    from fishstick.audio_separation.losses import (
        SeparationLoss,
        SISDRLoss,
        SDRLoss,
        PITLoss,
        DeepClusteringLoss,
        CompositeLoss,
    )
except ImportError:
    pass

try:
    from fishstick.audio_separation.augmentation import (
        AudioAugmentation,
        SpecAugment,
        RoomImpulseResponse,
        TimeStretch,
        PitchShift,
    )
except ImportError:
    pass

try:
    from fishstick.audio_separation.metrics import (
        SeparationMetrics,
        SI_SDR,
        SI_SAR,
        SI_SNR,
        PESQ,
        STOI,
        BSSEval,
    )
except ImportError:
    pass

try:
    from fishstick.audio_separation.training import (
        SeparationTrainer,
        SeparationDataset,
        create_dataloader,
    )
except ImportError:
    pass

try:
    from fishstick.audio_separation.preprocessing import (
        Preprocessor,
        Normalizer,
        AudioLoader,
        VoiceFilter,
    )
except ImportError:
    pass

try:
    from fishstick.audio_separation.postprocessing import (
        Postprocessor,
        AudioMerger,
        SourceFilter,
    )
except ImportError:
    pass

try:
    from fishstick.audio_separation.blocks import (
        ConvBlock,
        TransformerBlock,
        LSTMBlock,
        AttentionBlock,
    )
except ImportError:
    pass

__all__ = [
    "SeparationResult",
    "SeparationModel",
    "Separator",
    "STFT",
    "SeparationMetrics",
    "AudioMixer",
    "TimeFrequencyMask",
    "IdealBinaryMask",
    "IdealRatioMask",
    "PhaseSensitiveMask",
    "ComplexMask",
    "WienerFilter",
    "TFMaskingNetwork",
    "IBMEstimator",
    "DeepClustering",
    "EmbeddingExtractor",
    "ClusterAssigner",
    "DiscriminativeLoss",
    "WaveformSeparation",
    "ConvTasNet",
    "DualPathRNN",
    "TemporalConvNet",
    "SkipConnection",
    "SpeakerSeparator",
    "SpeakerBeam",
    "SpeakerRecognition",
    "VoiceActivityDetector",
    "MusicSeparator",
    "Demucs",
    "OpenUnmix",
    "XUMX",
    "BandSplit",
]
