"""
SSL Extensions Module for Fishstick

Comprehensive self-supervised learning extensions including:
- BYOL/SimSiam implementations
- Masked prediction methods
- Clustering-based SSL
- Multi-modal SSL (CLIP-style)
- Advanced SSL projection heads
- Advanced loss functions
"""

from fishstick.ssl_extensions.base import (
    MomentumUpdater,
    EMAUpdater,
    MemoryBank,
    StopGradient,
    stop_gradient,
    gather_from_all,
    SSLScheduler,
    BatchNorm1dSync,
    L2Normalize,
    MultiCropWrapper,
    DropPath,
    Patchify,
    Unpatchify,
    PositionalEmbedding2D,
    get_2d_sincos_pos_embed,
)

from fishstick.ssl_extensions.byol_simsiam import (
    BYOLLoss,
    AdvancedBYOL,
    SimSiamLoss,
    AdvancedSimSiam,
    MoCoV3,
    NNCLR,
)

from fishstick.ssl_extensions.masked_prediction import (
    MaskGenerator,
    MaskedImageModeling,
    Data2VecMIM,
    TokenLabeling,
    AudioMaskedPrediction,
    VideoMaskedPrediction,
)

from fishstick.ssl_extensions.clustering_ssl import (
    DeepCluster,
    SwAV,
    PrototypicalContrastive,
    SCANLoss,
    SCAN,
    OnlineKMeans,
)

from fishstick.ssl_extensions.multimodal_ssl import (
    CLIPTextEncoder,
    CLIPImageEncoder,
    CLIPLoss,
    CLIPModel,
    AudioEncoder,
    AudioVisualSSL,
    CrossModalRetriever,
    MultiModalProjector,
    ALIGNModel,
)

from fishstick.ssl_extensions.projection_heads import (
    MLProjectionHead,
    TransformerProjectionHead,
    MultiLayerProjectionHead,
    CosineProjectionHead,
    NonLinearProjectionHead,
    BYOLProjectionHead,
    SimSiamProjectionHead,
    SimSiamPredictorHead,
    SwAVProjectionHead,
    TemporalProjectionHead,
    MemoryBankProjection,
    ProjectionHeadEnsemble,
    LinearProjectionHead,
    IdentityProjectionHead,
)

from fishstick.ssl_extensions.losses import (
    NTXentLoss,
    SimSiamContrastiveLoss,
    VICRegLoss,
    BarlowTwinsLoss,
    DINOLoss,
    WMSELoss,
    DebiasedContrastiveLoss,
    HardNegativeContrastiveLoss,
    SupConLoss,
    TripletLoss,
    CenterLoss,
    ClusterLoss,
    RegularizationLoss,
)

__all__ = [
    "MomentumUpdater",
    "EMAUpdater",
    "MemoryBank",
    "StopGradient",
    "stop_gradient",
    "gather_from_all",
    "SSLScheduler",
    "BatchNorm1dSync",
    "L2Normalize",
    "MultiCropWrapper",
    "DropPath",
    "Patchify",
    "Unpatchify",
    "PositionalEmbedding2D",
    "get_2d_sincos_pos_embed",
    "BYOLLoss",
    "AdvancedBYOL",
    "SimSiamLoss",
    "AdvancedSimSiam",
    "MoCoV3",
    "NNCLR",
    "MaskGenerator",
    "MaskedImageModeling",
    "Data2VecMIM",
    "TokenLabeling",
    "AudioMaskedPrediction",
    "VideoMaskedPrediction",
    "DeepCluster",
    "SwAV",
    "PrototypicalContrastive",
    "SCANLoss",
    "SCAN",
    "OnlineKMeans",
    "CLIPTextEncoder",
    "CLIPImageEncoder",
    "CLIPLoss",
    "CLIPModel",
    "AudioEncoder",
    "AudioVisualSSL",
    "CrossModalRetriever",
    "MultiModalProjector",
    "ALIGNModel",
    "MLProjectionHead",
    "TransformerProjectionHead",
    "MultiLayerProjectionHead",
    "CosineProjectionHead",
    "NonLinearProjectionHead",
    "BYOLProjectionHead",
    "SimSiamProjectionHead",
    "SimSiamPredictorHead",
    "SwAVProjectionHead",
    "TemporalProjectionHead",
    "MemoryBankProjection",
    "ProjectionHeadEnsemble",
    "LinearProjectionHead",
    "IdentityProjectionHead",
    "NTXentLoss",
    "SimSiamContrastiveLoss",
    "VICRegLoss",
    "BarlowTwinsLoss",
    "DINOLoss",
    "WMSELoss",
    "DebiasedContrastiveLoss",
    "HardNegativeContrastiveLoss",
    "SupConLoss",
    "TripletLoss",
    "CenterLoss",
    "ClusterLoss",
    "RegularizationLoss",
]
