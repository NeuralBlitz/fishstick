"""
Medical Imaging Module

Comprehensive medical imaging tools for the fishstick AI framework.

This module provides:
- Medical image segmentation (U-Net, nnU-Net, loss functions)
- Radiology-specific transforms (window/level, HU normalization)
- 3D medical image processing (volumetric operations, resampling)
- Medical image registration (affine, deformable, VoxelMorph)
- Disease classification frameworks (CheXpert, chest X-ray models)

Author: fishstick AI Framework
Version: 0.2.0
"""

from fishstick.medical_imaging.segmentation import (
    nnUNet,
    nnUNetConfiguration,
    nnUNetEncoder,
    nnUNetDecoder,
    DeepSupervisionHead,
    TopKPathAggregation,
    UNet3D,
    ResidualUNet3D,
    AttentionUNet3D,
    MedicalTransform,
    WindowLevelTransform,
    HUNormalize,
    CTBoneRemoval,
    MRNormalize,
    OrganSegmentationModel,
    LiverSegmentationModel,
    LungSegmentationModel,
    BrainTumorSegmentationModel,
    DiceLoss,
    TverskyLoss,
    FocalDiceLoss,
    DiceScore,
    IoUScore,
    ConnectedComponentsPostProcessor,
    TestTimeAugmentationSegmentor,
)

from fishstick.medical_imaging.classification import (
    ResNet3DMedical,
    DenseNet3DMedical,
    CheXpertModel,
    ChestXrayClassifier,
    MedicalBackbone,
    MultiTaskClassifier,
    EnsembleClassifier,
    MILPool,
    AttentionMIL,
    DiseaseClassifier,
    CheXpertLabel,
    generate_chexpert_labels,
)

from fishstick.medical_imaging.registration import (
    VoxelMorph,
    VoxelMorphLoss,
    AffineRegistration,
    DeformableRegistration,
    DemonsRegistration,
    SymmetricNormalization,
    RegistrationResult,
    compute_similarity_metric,
    ncc_loss,
    mse_loss,
    dice_score_registration,
    apply_transform,
    compose_transforms,
    compute_jacobian_determinant,
)

__version__ = "0.2.0"

__all__ = [
    # Segmentation
    "nnUNet",
    "nnUNetConfiguration",
    "nnUNetEncoder",
    "nnUNetDecoder",
    "DeepSupervisionHead",
    "TopKPathAggregation",
    "UNet3D",
    "ResidualUNet3D",
    "AttentionUNet3D",
    "MedicalTransform",
    "WindowLevelTransform",
    "HUNormalize",
    "CTBoneRemoval",
    "MRNormalize",
    "OrganSegmentationModel",
    "LiverSegmentationModel",
    "LungSegmentationModel",
    "BrainTumorSegmentationModel",
    "DiceLoss",
    "TverskyLoss",
    "FocalDiceLoss",
    "DiceScore",
    "IoUScore",
    "ConnectedComponentsPostProcessor",
    "TestTimeAugmentationSegmentor",
    # Classification
    "ResNet3DMedical",
    "DenseNet3DMedical",
    "CheXpertModel",
    "ChestXrayClassifier",
    "MedicalBackbone",
    "MultiTaskClassifier",
    "EnsembleClassifier",
    "MILPool",
    "AttentionMIL",
    "DiseaseClassifier",
    "CheXpertLabel",
    "generate_chexpert_labels",
    # Registration
    "VoxelMorph",
    "VoxelMorphLoss",
    "AffineRegistration",
    "DeformableRegistration",
    "DemonsRegistration",
    "SymmetricNormalization",
    "RegistrationResult",
    "compute_similarity_metric",
    "ncc_loss",
    "mse_loss",
    "dice_score_registration",
    "apply_transform",
    "compose_transforms",
    "compute_jacobian_determinant",
]
