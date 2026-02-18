"""
Multimodal Advanced Module for Fishstick AI Framework.

Advanced multimodal learning implementations including:
- CLIP: Contrastive Language-Image Pre-Training
- Audio-Visual: Cross-modal attention and audio-visual correspondence
- Fusion: Early fusion, late fusion, cross-attention, Perceiver Resampler
"""

from .clip import (
    CLIPConfig,
    ImageEncoder,
    TextEncoder,
    CLIPModel,
    clip_loss,
    CLIPWithProjection,
    DEFAULT_CLIP_CONFIG,
)

from .align import (
    AudioVisualConfig,
    CrossModalAttention,
    AudioEncoder,
    VideoEncoder,
    AudioVisualEncoder,
    SyncNet,
    AudioVisualCorrespondence,
    contrastive_alignment_loss,
    syncnet_contrastive_loss,
    DEFAULT_AUDIO_VISUAL_CONFIG,
)

from .fusion import (
    FusionConfig,
    EarlyFusion,
    LateFusion,
    CrossAttentionFusion,
    PerceiverResampler,
    MultimodalPerceiverResampler,
    GatedFusion,
    FiLMFusion,
    create_fusion_module,
    DEFAULT_FUSION_CONFIG,
)


__all__ = [
    # CLIP
    "CLIPConfig",
    "ImageEncoder",
    "TextEncoder",
    "CLIPModel",
    "clip_loss",
    "CLIPWithProjection",
    "DEFAULT_CLIP_CONFIG",
    # Audio-Visual
    "AudioVisualConfig",
    "CrossModalAttention",
    "AudioEncoder",
    "VideoEncoder",
    "AudioVisualEncoder",
    "SyncNet",
    "AudioVisualCorrespondence",
    "contrastive_alignment_loss",
    "syncnet_contrastive_loss",
    "DEFAULT_AUDIO_VISUAL_CONFIG",
    # Fusion
    "FusionConfig",
    "EarlyFusion",
    "LateFusion",
    "CrossAttentionFusion",
    "PerceiverResampler",
    "MultimodalPerceiverResampler",
    "GatedFusion",
    "FiLMFusion",
    "create_fusion_module",
    "DEFAULT_FUSION_CONFIG",
]
