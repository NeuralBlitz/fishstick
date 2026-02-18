"""
fishstick augmentation_ext
==========================

Extended data augmentation tools for the fishstick AI framework.

This module provides comprehensive data augmentation capabilities across multiple data modalities:
- Image augmentation (advanced MixUp, CutMix, RandAugment, AutoAugment, etc.)
- Video augmentation (temporal augmentations, 3D transforms)
- Tabular data augmentation (SMOTE, noise injection, feature shuffling)
- Graph augmentation (node/edge dropping, attribute masking)
- Audio augmentation (time stretching, pitch shifting, SpecAugment)

Based on the UniIntelli framework with categorical-geometric-thermodynamic synthesis.

Author: fishstick AI Framework
Version: 0.1.0
"""

from fishstick.augmentation_ext.base import (
    AugmentationBase,
    AugmentationConfig,
    AugmentationScheduler,
    AdaptiveAugmentation,
    AugmentationPipeline,
    MixupCutmixCollator,
    get_augmentation_pipeline,
)

from fishstick.augmentation_ext.image_augmentation import (
    RandomHorizontalFlip,
    RandomVerticalFlip,
    RandomRotation,
    ColorJitter,
    RandomAffine,
    MixUp,
    CutMix,
    GridMask,
    RandomErasing,
    Mosaic,
    Blend,
    RandAugment,
    AutoAugment,
    GaussianBlur,
    Cutout,
    get_image_augmentation_pipeline,
)

from fishstick.augmentation_ext.video_augmentation import (
    TemporalDropout,
    RandomCropResize,
    ColorJitterVideo,
    RandomRotation3D,
    MixUpVideo,
    CutMixVideo,
    FrameShuffle,
    TemporalReverse,
    RandomSpeed,
    VideoSpatialDropout,
    TemporalJitter,
    get_video_augmentation_pipeline,
)

from fishstick.augmentation_ext.tabular_augmentation import (
    SMOTE,
    RandomNoiseInjection,
    FeatureShuffle,
    RowMixing,
    SMOTETomek,
    ADASYN,
    FeatureNoiseMask,
    CutoffAugmentation,
    SwapNoise,
    cGANAugmentation,
    get_tabular_augmentation_pipeline,
)

from fishstick.augmentation_ext.graph_augmentation import (
    GraphData,
    NodeDrop,
    EdgeDrop,
    AttributeMasking,
    SubgraphExtraction,
    NodeFeatureNoise,
    EdgeWeightPerturbation,
    GraphMixup,
    PersonalizedPageRank,
    GraphDataAugmentation,
    get_graph_augmentation_pipeline,
)

from fishstick.augmentation_ext.audio_augmentation import (
    TimeStretch,
    PitchShift,
    AddBackgroundNoise,
    TimeShift,
    VolumePerturbation,
    SpecAugment,
    AudioMixUp,
    TimeCrop,
    AudioSpeed,
    ReverbEffect,
    get_audio_augmentation_pipeline,
)

from fishstick.augmentation_ext.pipeline import (
    AugmentationType,
    AugmentationInfo,
    ConditionalAugmentation,
    AugmentationCache,
    AugmentationSequence,
    AugmentationEnsemble,
    AugmentationMixer,
    MultiModalAugmentation,
    ProgressiveAugmentation,
    AugmentationFactory,
    create_standard_pipeline,
)

__version__ = "0.1.0"

__all__ = [
    "AugmentationBase",
    "AugmentationConfig",
    "AugmentationScheduler",
    "AdaptiveAugmentation",
    "AugmentationPipeline",
    "MixupCutmixCollator",
    "get_augmentation_pipeline",
    "RandomHorizontalFlip",
    "RandomVerticalFlip",
    "RandomRotation",
    "ColorJitter",
    "RandomAffine",
    "MixUp",
    "CutMix",
    "GridMask",
    "RandomErasing",
    "Mosaic",
    "Blend",
    "RandAugment",
    "AutoAugment",
    "GaussianBlur",
    "Cutout",
    "get_image_augmentation_pipeline",
    "TemporalDropout",
    "RandomCropResize",
    "ColorJitterVideo",
    "RandomRotation3D",
    "MixUpVideo",
    "CutMixVideo",
    "FrameShuffle",
    "TemporalReverse",
    "RandomSpeed",
    "VideoSpatialDropout",
    "TemporalJitter",
    "get_video_augmentation_pipeline",
    "SMOTE",
    "RandomNoiseInjection",
    "FeatureShuffle",
    "RowMixing",
    "SMOTETomek",
    "ADASYN",
    "FeatureNoiseMask",
    "CutoffAugmentation",
    "SwapNoise",
    "cGANAugmentation",
    "get_tabular_augmentation_pipeline",
    "GraphData",
    "NodeDrop",
    "EdgeDrop",
    "AttributeMasking",
    "SubgraphExtraction",
    "NodeFeatureNoise",
    "EdgeWeightPerturbation",
    "GraphMixup",
    "PersonalizedPageRank",
    "GraphDataAugmentation",
    "get_graph_augmentation_pipeline",
    "TimeStretch",
    "PitchShift",
    "AddBackgroundNoise",
    "TimeShift",
    "VolumePerturbation",
    "SpecAugment",
    "AudioMixUp",
    "TimeCrop",
    "AudioSpeed",
    "ReverbEffect",
    "get_audio_augmentation_pipeline",
    "AugmentationType",
    "AugmentationInfo",
    "ConditionalAugmentation",
    "AugmentationCache",
    "AugmentationSequence",
    "AugmentationEnsemble",
    "AugmentationMixer",
    "MultiModalAugmentation",
    "ProgressiveAugmentation",
    "AugmentationFactory",
    "create_standard_pipeline",
]
