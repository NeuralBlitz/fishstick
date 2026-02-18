"""
Multi-Modal & Cross-Modal Learning Extensions for fishstick

This module provides comprehensive tools for multi-modal and cross-modal learning:
- Vision-language models (CLIP, BLIP)
- Audio-visual learning
- Cross-modal retrieval
- Multi-modal fusion
- Modality alignment

All modules follow fishstick's coding standards with:
- Comprehensive docstrings
- Type hints throughout
- PyTorch nn.Module base classes
- Factory functions for easy model creation
"""

from typing import Optional

from .vision_language import (
    CLIPVisionEncoder,
    CLIPTextEncoder,
    CLIPModel,
    BLIPImageEncoder,
    BLIPTextEncoder,
    BLIPModel,
    VisionLanguageTransformer,
    create_clip_model,
    create_blip_model,
)

from .image_text_matching import (
    DualEncoderMatcher,
    CrossAttentionMatcher,
    SimilarityMatrix,
    RankingLoss,
    ImageTextRetrieval,
    HardNegativeMiner,
    create_matcher,
)

from .visual_question_answering import (
    QuestionEncoder,
    BottomUpAttention,
    SAN,
    VQAModel,
    VQAWithAttention,
    LXMERTStyleEncoder,
    create_vqa_model,
)

from .audio_visual import (
    AudioEncoder2D,
    VideoEncoder,
    AudioVisualCorrespondence,
    AudioVisualEventDetector,
    SoundLocalization,
    AudioVisualSync,
    ContrastiveAudioVisualLoss,
    create_audiovisual_model,
)

from .audio_visual_fusion import (
    AVFuser,
    EarlyFusionAV,
    LateFusionAV,
    CrossModalFusion,
    TensorFusion,
    GatedFusionAV,
    MemoryAugmentedFusion,
    FiLMFusion,
    create_av_fusion,
)

from .cross_modal_retrieval import (
    DualEncoderRetriever,
    CrossModalHasher,
    RetrievalRanker,
    SemanticRetriever,
    CrossModalAttentionRetrieval,
    RetrievalLoss,
    CMRLoss,
    HardNegativeMining,
    create_retriever,
)

from .contrastive_learning import (
    MultiModalSimCLR,
    SimCLRLoss,
    CLIPLoss,
    TripletContrastiveLoss,
    MoCoStyleLoss,
    InfoNCELoss,
    BarlowTwinsLoss,
    MultiModalContrastiveModel,
    HardNegativeContrastiveLoss,
    create_contrastive_loss,
)

from .attention_fusion import (
    CrossModalAttention,
    SelfAttentionFusion,
    CoAttentionFusion,
    LowRankBilinearAttention,
    MultiHeadBilinearFusion,
    TFN,
    MemoryAttentionFusion,
    StackedAttentionFusion,
    create_attention_fusion,
)

from .tensor_fusion import (
    TensorFusionNetwork,
    FactorizedBilinearNetwork,
    MultimodalFactorizedBilinear,
    LowRankTensorFusion,
    HadamardFusion,
    ConcatenationFusion,
    GMU,
    MultimodalBottleneck,
    MixFusion,
    create_tensor_fusion,
)

from .adversarial_alignment import (
    GradientReversal,
    DomainDiscriminator,
    AdversarialModalityAligner,
    CycleConsistentAlignment,
    MMDAlignment,
    CoralAlignment,
    MultiModalAdversarialNetwork,
    DANNLoss,
    create_alignment_module,
)

from .optimal_transport import (
    SinkhornDistance,
    WassersteinDistance,
    OptimalTransportAlignment,
    EntropicOT,
    UnbalancedOT,
    POTAlignment,
    GromovWasserstein,
    Sinkhornknopp,
    NeuralOptimalTransport,
    create_ot_module,
)

__all__ = [
    "CLIPVisionEncoder",
    "CLIPTextEncoder",
    "CLIPModel",
    "BLIPImageEncoder",
    "BLIPTextEncoder",
    "BLIPModel",
    "VisionLanguageTransformer",
    "create_clip_model",
    "create_blip_model",
    "DualEncoderMatcher",
    "CrossAttentionMatcher",
    "SimilarityMatrix",
    "RankingLoss",
    "ImageTextRetrieval",
    "HardNegativeMiner",
    "create_matcher",
    "QuestionEncoder",
    "BottomUpAttention",
    "SAN",
    "VQAModel",
    "VQAWithAttention",
    "LXMERTStyleEncoder",
    "create_vqa_model",
    "AudioEncoder2D",
    "VideoEncoder",
    "AudioVisualCorrespondence",
    "AudioVisualEventDetector",
    "SoundLocalization",
    "AudioVisualSync",
    "ContrastiveAudioVisualLoss",
    "create_audiovisual_model",
    "AVFuser",
    "EarlyFusionAV",
    "LateFusionAV",
    "CrossModalFusion",
    "TensorFusion",
    "GatedFusionAV",
    "MemoryAugmentedFusion",
    "FiLMFusion",
    "create_av_fusion",
    "DualEncoderRetriever",
    "CrossModalHasher",
    "RetrievalRanker",
    "SemanticRetriever",
    "CrossModalAttentionRetrieval",
    "RetrievalLoss",
    "CMRLoss",
    "HardNegativeMining",
    "create_retriever",
    "MultiModalSimCLR",
    "SimCLRLoss",
    "CLIPLoss",
    "TripletContrastiveLoss",
    "MoCoStyleLoss",
    "InfoNCELoss",
    "BarlowTwinsLoss",
    "MultiModalContrastiveModel",
    "HardNegativeContrastiveLoss",
    "create_contrastive_loss",
    "CrossModalAttention",
    "SelfAttentionFusion",
    "CoAttentionFusion",
    "LowRankBilinearAttention",
    "MultiHeadBilinearFusion",
    "TFN",
    "MemoryAttentionFusion",
    "StackedAttentionFusion",
    "create_attention_fusion",
    "TensorFusionNetwork",
    "FactorizedBilinearNetwork",
    "MultimodalFactorizedBilinear",
    "LowRankTensorFusion",
    "HadamardFusion",
    "ConcatenationFusion",
    "GMU",
    "MultimodalBottleneck",
    "MixFusion",
    "create_tensor_fusion",
    "GradientReversal",
    "DomainDiscriminator",
    "AdversarialModalityAligner",
    "CycleConsistentAlignment",
    "MMDAlignment",
    "CoralAlignment",
    "MultiModalAdversarialNetwork",
    "DANNLoss",
    "create_alignment_module",
    "SinkhornDistance",
    "WassersteinDistance",
    "OptimalTransportAlignment",
    "EntropicOT",
    "UnbalancedOT",
    "POTAlignment",
    "GromovWasserstein",
    "Sinkhornknopp",
    "NeuralOptimalTransport",
    "create_ot_module",
]


def create_multimodal_model(
    model_type: str,
    **kwargs,
):
    """
    Factory function to create multi-modal models.

    Args:
        model_type: Type of model to create
        **kwargs: Additional arguments for the model

    Returns:
        Multi-modal model module
    """
    if model_type == "clip":
        return create_clip_model(**kwargs)
    elif model_type == "blip":
        return create_blip_model(**kwargs)
    elif model_type == "vqa":
        return create_vqa_model(**kwargs)
    elif model_type == "retrieval":
        return create_retriever(**kwargs)
    elif model_type == "contrastive":
        return create_contrastive_loss(**kwargs)
    elif model_type == "fusion_attention":
        return create_attention_fusion(**kwargs)
    elif model_type == "fusion_tensor":
        return create_tensor_fusion(**kwargs)
    elif model_type == "alignment_adversarial":
        return create_alignment_module("adversarial", **kwargs)
    elif model_type == "alignment_ot":
        return create_ot_module(**kwargs)
    else:
        raise ValueError(f"Unknown model type: {model_type}")


__all__.append("create_multimodal_model")
