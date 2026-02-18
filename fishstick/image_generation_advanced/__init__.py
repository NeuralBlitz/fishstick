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

_3d_gen = importlib.import_module(".3d_generation", __package__)

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

from . import _3d_gen as gen3d

from ._3d_generation import (
    VoxelGenerator3D,
    VoxelDiscriminator3D,
    GANomaly3D,
    NeuralRadianceField,
    EG3DGenerator,
    SynthesisNetwork,
    StyleBlock,
    ToRGB,
    ToDepth,
    PointCloudGenerator,
    PointNetDiscriminator,
    TriplaneGenerator,
    CameraEncoder,
    GaussianSplatting,
    PointCloudToImage,
)

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
    "StyleBlock",
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
