"""
Self-Supervised Learning Module

Comprehensive self-supervised learning implementations including:
- Contrastive learning (SimCLR, BYOL, SimSiam, MoCo)
- Masked autoencoders (MAE, SimMIM)
- Deep InfoMax
- Barlow Twins
"""

from fishstick.selfsupervised.contrastive import (
    SimCLR,
    BYOL,
    SimSiam,
    MoCo,
    ContrastiveHead,
)
from fishstick.selfsupervised.masked import (
    MAE,
    SimMIM,
    MaskedAutoencoder,
    PatchEmbed,
)
from fishstick.selfsupervised.deep_infomax import (
    DeepInfoMax,
    GlobalInfoMax,
    LocalInfoMax,
)
from fishstick.selfsupervised.barlow_twins import (
    BarlowTwins,
    BarlowTwinsLoss,
)
from fishstick.selfsupervised.losses import (
    NT_XentLoss,
    SimSiamLoss,
    BYOLLoss,
    MoCoLoss,
    VicRegLoss,
    InfoNCE,
)
from fishstick.selfsupervised.augmentations import (
    BYOLAugmentations,
    SimCLRAugmentations,
    MAEAugmentations,
    RandomResizedCrop,
    ColorJitter,
    GaussianBlur,
    Solarization,
)

__all__ = [
    "SimCLR",
    "BYOL",
    "SimSiam",
    "MoCo",
    "ContrastiveHead",
    "MAE",
    "SimMIM",
    "MaskedAutoencoder",
    "PatchEmbed",
    "DeepInfoMax",
    "GlobalInfoMax",
    "LocalInfoMax",
    "BarlowTwins",
    "BarlowTwinsLoss",
    "NT_XentLoss",
    "SimSiamLoss",
    "BYOLLoss",
    "MoCoLoss",
    "VicRegLoss",
    "InfoNCE",
    "BYOLAugmentations",
    "SimCLRAugmentations",
    "MAEAugmentations",
    "RandomResizedCrop",
    "ColorJitter",
    "GaussianBlur",
    "Solarization",
]
