"""
Generative Models for fishstick AI Framework.

Implements advanced generative model architectures:
- Diffusion Models: DDPM, DDIM, Score-based, Latent Diffusion
- GAN Extensions: StyleGAN, BigGAN, Projection Discriminator
- Autoregressive Transformers: GPT, PixelCNN
- Energy-Based Models: EBMs with sampling
- Flow Matching: OT-Flow, Conditional Flow Matching

References:
- DDPM: Ho et al. (2020) "Denoising Diffusion Probabilistic Models"
- DDIM: Song et al. (2021) "Denoising Diffusion Implicit Models"
- Score-Based: Song & Ermon (2019) "Generative Modeling by Estimating Gradients"
- Latent Diffusion: Rombach et al. (2022) "High-Resolution Image Synthesis with Latent Diffusion Models"
- StyleGAN: Karras et al. (2019) "A Style-Based Generator Architecture for GANs"
- BigGAN: Brock et al. (2019) "Large Scale GAN Training for High Fidelity Natural Image Synthesis"
- Flow Matching: Lipman et al. (2023) "Flow Matching for Generative Modeling"
"""

from typing import Optional, Tuple, List, Dict, Callable, Union
import torch
from torch import Tensor, nn

from .ddpm import DDPM, DiffusionScheduler
from .ddim import DDIM, DDIMScheduler
from .score_based import ScoreBasedModel, ScoreNetwork, AnnealedLangevinDynamics
from .latent_diffusion import LatentDiffusionModel, AutoencoderKL, VQModel
from .stylegan import StyleGAN, StyleGAN2, MappingNetwork, SynthesisNetwork
from .stylegan import StyleGAN2Generator, ProgressiveGrowing
from .biggan import BigGAN, ConditionalGenerator, ProjectionDiscriminator
from .autoregressive_transformer import AutoregressiveTransformer, PositionalEncoding
from .gpt_generation import GPTGenerator, GPT2LMHeadModel
from .pixel_cnn import PixelCNN, PixelCNNPP, GatedMaskedConv2d
from .energy_model import EnergyBasedModel, ConvEnergyModel
from .ebm_sampling import LangevinSampler, HMCSampler, EBMTrainer
from .flow_matching import FlowMatching, ConditionalFlowMatching
from .ot_flow import OptimalTransportFlow, SinkhornDivergence

__all__ = [
    "DDPM",
    "DiffusionScheduler",
    "DDIM",
    "DDIMScheduler",
    "ScoreBasedModel",
    "ScoreNetwork",
    "AnnealedLangevinDynamics",
    "LatentDiffusionModel",
    "AutoencoderKL",
    "VQModel",
    "StyleGAN",
    "StyleGAN2",
    "MappingNetwork",
    "SynthesisNetwork",
    "StyleGAN2Generator",
    "ProgressiveGrowing",
    "BigGAN",
    "ConditionalGenerator",
    "ProjectionDiscriminator",
    "AutoregressiveTransformer",
    "PositionalEncoding",
    "GPTGenerator",
    "GPT2LMHeadModel",
    "PixelCNN",
    "PixelCNNPP",
    "GatedMaskedConv2d",
    "EnergyBasedModel",
    "ConvEnergyModel",
    "LangevinSampler",
    "HMCSampler",
    "EBMTrainer",
    "FlowMatching",
    "ConditionalFlowMatching",
    "OptimalTransportFlow",
    "SinkhornDivergence",
]

_GENERATIVE_AVAILABLE = True
