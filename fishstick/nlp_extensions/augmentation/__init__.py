"""
Augmentation Module

Text augmentation techniques including back translation, synonym replacement, and more.
"""

from fishstick.nlp_extensions.augmentation.back_translation import (
    BackTranslationAugmentation,
    MultilingualBackTranslation,
    RoundTripTranslation,
    BackTranslationWithNoise,
)
from fishstick.nlp_extensions.augmentation.synonym_replacement import (
    SynonymReplacement,
    EasyDataAugmentation,
    ContextualSynonymReplacement,
)
from fishstick.nlp_extensions.augmentation.random_edits import (
    RandomInsertion,
    RandomDeletion,
    RandomSwap,
    TextAugmenter,
    CharacterLevelAugmentation,
    BackRandomEdits,
)
from fishstick.nlp_extensions.augmentation.contextual_augmentation import (
    ContextualAugmentation,
    ConditionalContextualAugmentation,
    BERTContextualAugmentation,
    FillMaskAugmentation,
)

__all__ = [
    "BackTranslationAugmentation",
    "MultilingualBackTranslation",
    "RoundTripTranslation",
    "BackTranslationWithNoise",
    "SynonymReplacement",
    "EasyDataAugmentation",
    "ContextualSynonymReplacement",
    "RandomInsertion",
    "RandomDeletion",
    "RandomSwap",
    "TextAugmenter",
    "CharacterLevelAugmentation",
    "BackRandomEdits",
    "ContextualAugmentation",
    "ConditionalContextualAugmentation",
    "BERTContextualAugmentation",
    "FillMaskAugmentation",
]
