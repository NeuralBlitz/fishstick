"""
Advanced Image Generation Module.

This module provides advanced image generation models:
- Flow-based models (Glow, RealNVP, Flow++)
- Energy-Based Models (EBMs, Langevin dynamics)
- Text-to-Image models (CLIP-based, diffusion-based)
- Super-Resolution models (ESRGAN, SRFlow, SwinIR)
- 3D-Aware Generation (NeRF, EG3D, Point Clouds)
- Inpainting models (DeepFill, EdgeConnect, Diffusion)
"""

import importlib
from typing import Any

_3d_module = importlib.import_module(".3d_generation", __package__)

from .flows import (
    ActNorm,
    Invertible1x1Conv,
    AffineCoupling,
    FlowStep,
    Squeeze,
    Glow,
    RealNVP,
    RealNVPCoupling,
    MaskedLinear,
    VariationalDequantization,
    FlowPlusPlus,
)

from .energy_models import (
    EnergyFunction,
    ConvolutionalEnergyFunction,
    ConditionalEnergyFunction,
    EBM,
    ConditionalEBM,
    MCMCScheduler,
    DenoisingEBM,
    GradientPenaltyEBM,
    JointEBM,
)

from .text_to_image import (
    TextEncoder,
    CLIPTextEncoder,
    TextConditioning,
    CrossAttention,
    TextToImageDiffusion,
    PromptEmbedder,
    ClassifierFreeGuidance,
    MultiPromptGenerator,
    ImageVariationGenerator,
    StableDiffusionXL,
    VAEEncoderDecoder,
    TimeEmbedding,
)

from .super_resolution import (
    ResidualDenseBlock,
    ResidualInResidualDenseBlock,
    ESRGANGenerator,
    ESRGANDiscriminator,
    SRFlow,
    FlowBlock,
    ActNorm2d,
    InvConv2d,
    AffineCoupling2d,
    SwinIRBlock,
    SwinIR,
    RealESRGAN,
    EDVRNet,
    PCDAalignment,
    RCANBlock,
    RCAN,
)

VoxelGenerator3D = getattr(_3d_module, "VoxelGenerator3D")
VoxelDiscriminator3D = getattr(_3d_module, "VoxelDiscriminator3D")
GANomaly3D = getattr(_3d_module, "GANomaly3D")
NeuralRadianceField = getattr(_3d_module, "NeuralRadianceField")
EG3DGenerator = getattr(_3d_module, "EG3DGenerator")
SynthesisNetwork = getattr(_3d_module, "SynthesisNetwork")
StyleBlock3D = getattr(_3d_module, "StyleBlock")
ToRGB = getattr(_3d_module, "ToRGB")
ToDepth = getattr(_3d_module, "ToDepth")
PointCloudGenerator = getattr(_3d_module, "PointCloudGenerator")
PointNetDiscriminator = getattr(_3d_module, "PointNetDiscriminator")
TriplaneGenerator = getattr(_3d_module, "TriplaneGenerator")
CameraEncoder = getattr(_3d_module, "CameraEncoder")
GaussianSplatting = getattr(_3d_module, "GaussianSplatting")
PointCloudToImage = getattr(_3d_module, "PointCloudToImage")

from .inpainting import (
    GatedConv,
    ContextualAttention,
    DeepFillGenerator,
    EdgeGenerator,
    ImageInpaintingNet,
    EdgeConnect,
    DiffusionInpainting,
    GuidedFilter,
    LearnableBlur,
    ImageHarmonizationNet,
    MaskPredictionNet,
    PartialConv,
    PartialConvInpainting,
    MAT,
)

__all__ = [
    "ActNorm",
    "Invertible1x1Conv",
    "AffineCoupling",
    "FlowStep",
    "Squeeze",
    "Glow",
    "RealNVP",
    "RealNVPCoupling",
    "MaskedLinear",
    "VariationalDequantization",
    "FlowPlusPlus",
    "EnergyFunction",
    "ConvolutionalEnergyFunction",
    "ConditionalEnergyFunction",
    "EBM",
    "ConditionalEBM",
    "MCMCScheduler",
    "DenoisingEBM",
    "GradientPenaltyEBM",
    "JointEBM",
    "TextEncoder",
    "CLIPTextEncoder",
    "TextConditioning",
    "CrossAttention",
    "TextToImageDiffusion",
    "PromptEmbedder",
    "ClassifierFreeGuidance",
    "MultiPromptGenerator",
    "ImageVariationGenerator",
    "StableDiffusionXL",
    "VAEEncoderDecoder",
    "TimeEmbedding",
    "ResidualDenseBlock",
    "ResidualInResidualDenseBlock",
    "ESRGANGenerator",
    "ESRGANDiscriminator",
    "SRFlow",
    "FlowBlock",
    "ActNorm2d",
    "InvConv2d",
    "AffineCoupling2d",
    "SwinIRBlock",
    "SwinIR",
    "RealESRGAN",
    "EDVRNet",
    "PCDAalignment",
    "RCANBlock",
    "RCAN",
    "VoxelGenerator3D",
    "VoxelDiscriminator3D",
    "GANomaly3D",
    "NeuralRadianceField",
    "EG3DGenerator",
    "SynthesisNetwork",
    "StyleBlock3D",
    "ToRGB",
    "ToDepth",
    "PointCloudGenerator",
    "PointNetDiscriminator",
    "TriplaneGenerator",
    "CameraEncoder",
    "GaussianSplatting",
    "PointCloudToImage",
    "GatedConv",
    "ContextualAttention",
    "DeepFillGenerator",
    "EdgeGenerator",
    "ImageInpaintingNet",
    "EdgeConnect",
    "DiffusionInpainting",
    "GuidedFilter",
    "LearnableBlur",
    "ImageHarmonizationNet",
    "MaskPredictionNet",
    "PartialConv",
    "PartialConvInpainting",
    "MAT",
]
