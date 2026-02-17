"""
fishstick Multi-Modal Learning Module

Multi-modal fusion and learning.
"""

from fishstick.multimodal.fusion import (
    EarlyFusion,
    LateFusion,
    CrossModalAttention,
    ModalityAlignment,
)
from fishstick.multimodal.encoder import (
    ImageEncoder,
    TextEncoder,
    AudioEncoder,
    MultiModalEncoder,
)

__all__ = [
    "EarlyFusion",
    "LateFusion",
    "CrossModalAttention",
    "ModalityAlignment",
    "ImageEncoder",
    "TextEncoder",
    "AudioEncoder",
    "MultiModalEncoder",
]
