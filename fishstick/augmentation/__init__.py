"""
Advanced Data Augmentation Module

State-of-the-art data augmentation techniques.
"""

from fishstick.augmentation.advanced import (
    Augmentation,
    CutOut,
    MixUp,
    CutMix,
    RandAugment,
    TrivialAugmentWide,
    AugmentationPipeline,
    MixupCutmixCollator,
    get_augmentation_pipeline,
)

from fishstick.augmentation.image_augments import (
    RandomCrop,
    RandomHorizontalFlip,
    RandomVerticalFlip,
    ColorJitter,
    RandomErasing,
    GaussianBlur,
    RandomRotation,
    Mixup,
    Cutmix,
    AutoAugment,
)

from fishstick.augmentation.text_augments import (
    SynonymReplacement,
    RandomInsertion,
    RandomDeletion,
    RandomSwap,
    BackTranslation,
    CharacterLevelAugment,
    WordDropout,
    EDA,
    create_text_augment,
)

from fishstick.augmentation.audio_augments import (
    AddNoise,
    TimeStretch,
    PitchShift,
    VolumeChange,
    SpecAugment,
    TimeShift,
    AudioCompose,
    create_audio_augment,
)

from fishstick.augmentation.schedule import (
    AugmentSchedule,
    ConstantSchedule,
    LinearSchedule,
    ExponentialSchedule,
    CosineAnnealingSchedule,
    PolicyScheduler,
    ScheduledAugment,
    AdaptiveProbabilityScheduler,
    RandomApply,
    create_schedule,
)

from fishstick.augmentation.compose import (
    Compose,
    OneOf,
    Sometimes,
    Repeat,
    Replay,
    Lambda,
    RandomChoice,
    Sequential,
    ApplyWithCondition,
    create_pipeline,
)

__all__ = [
    # advanced
    "Augmentation",
    "CutOut",
    "MixUp",
    "CutMix",
    "RandAugment",
    "TrivialAugmentWide",
    "AugmentationPipeline",
    "MixupCutmixCollator",
    "get_augmentation_pipeline",
    # image
    "RandomCrop",
    "RandomHorizontalFlip",
    "RandomVerticalFlip",
    "ColorJitter",
    "RandomErasing",
    "GaussianBlur",
    "RandomRotation",
    "Mixup",
    "Cutmix",
    "AutoAugment",
    # text
    "SynonymReplacement",
    "RandomInsertion",
    "RandomDeletion",
    "RandomSwap",
    "BackTranslation",
    "CharacterLevelAugment",
    "WordDropout",
    "EDA",
    "create_text_augment",
    # audio
    "AddNoise",
    "TimeStretch",
    "PitchShift",
    "VolumeChange",
    "SpecAugment",
    "TimeShift",
    "AudioCompose",
    "create_audio_augment",
    # schedule
    "AugmentSchedule",
    "ConstantSchedule",
    "LinearSchedule",
    "ExponentialSchedule",
    "CosineAnnealingSchedule",
    "PolicyScheduler",
    "ScheduledAugment",
    "AdaptiveProbabilityScheduler",
    "RandomApply",
    "create_schedule",
    # compose
    "Compose",
    "OneOf",
    "Sometimes",
    "Repeat",
    "Replay",
    "Lambda",
    "RandomChoice",
    "Sequential",
    "ApplyWithCondition",
    "create_pipeline",
]
