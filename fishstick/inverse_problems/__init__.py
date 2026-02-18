"""
Inverse Problems & Inverse Imaging Module

Comprehensive tools for solving inverse problems in imaging including:
- Compressed Sensing
- Image Deblurring
- Image Denoising
- MRI Reconstruction
- Tomography Reconstruction
- Regularization Techniques
- Iterative Solvers
"""

from .base import (
    LinearOperator,
    SensingMatrix,
    BlurKernel,
    InverseProblemLoss,
    psnr,
    ssim,
)

from .compressed_sensing import (
    CompressedSensingReconstructor,
    OMP,
    IHT,
    CoSaMP,
    TVMinimization,
    LearnedCompressedSensing,
    AdaptiveCompressedSensing,
    create_sensing_matrix,
    compute_coherence,
)

from .deblurring import (
    RichardsonLucyDeconvolution,
    WienerFilter,
    BlindDeblurring,
    DeepDeblurringNetwork,
    DeblurringWithMotionKernel,
    MultiScaleDeblurring,
    KernelEstimationNetwork,
    create_blur_kernel,
)

from .denoising import (
    NonLocalMeansDenoising,
    TotalVariationDenoising,
    BM3DDenoising,
    DnCNN,
    UNetDenoising,
    GaussianBlurDenoising,
    BilateralFilterDenoising,
    KSVMDenoising,
    DenoisingDiffusionModel,
    create_noisy_image,
)

from .mri_reconstruction import (
    MRIReconstructor,
    CompressedSensingMRI,
    SenseReconstruction,
    GrappaReconstruction,
    KspaceDeepReconstruction,
    VarNet,
    CoilSensitivityEstimator,
    RSSReconstruction,
    create_cartesian_mask,
    create_radial_mask,
    create_spiral_mask,
)

from .tomography import (
    TomographyReconstructor,
    FilteredBackProjection,
    ART,
    SIRT,
    DeepTomography,
    LearnedIterativeTomography,
    MLEM,
    create_sinogram,
)

from .regularization import (
    Regularizer,
    TikhonovRegularization,
    TotalVariationRegularization,
    L1Regularization,
    L0Regularization,
    ElasticNetRegularization,
    NuclearNormRegularization,
    SpectralRegularization,
    GroupLassoRegularization,
    LearnedRegularization,
    DeepPriorRegularization,
    HessianRegularization,
    WaveletRegularization,
    FusedRegularization,
    DirichletRegularization,
    HuberRegularization,
    ProximalOperator,
    create_regularizer,
)

from .solvers import (
    InverseProblemSolver,
    ConjugateGradient,
    ADMM,
    PrimalDualHybridGradient,
    GradientDescent,
    GaussNewton,
    ProximalGradientDescent,
    FISTA,
    SplitBregman,
    StochasticGradientDescent,
    LBFGS,
    create_solver,
)

__all__ = [
    # Base
    "LinearOperator",
    "SensingMatrix",
    "BlurKernel",
    "InverseProblemLoss",
    "psnr",
    "ssim",
    # Compressed Sensing
    "CompressedSensingReconstructor",
    "OMP",
    "IHT",
    "CoSaMP",
    "TVMinimization",
    "LearnedCompressedSensing",
    "AdaptiveCompressedSensing",
    "create_sensing_matrix",
    "compute_coherence",
    # Deblurring
    "RichardsonLucyDeconvolution",
    "WienerFilter",
    "BlindDeblurring",
    "DeepDeblurringNetwork",
    "DeblurringWithMotionKernel",
    "MultiScaleDeblurring",
    "KernelEstimationNetwork",
    "create_blur_kernel",
    # Denoising
    "NonLocalMeansDenoising",
    "TotalVariationDenoising",
    "BM3DDenoising",
    "DnCNN",
    "UNetDenoising",
    "GaussianBlurDenoising",
    "BilateralFilterDenoising",
    "KSVMDenoising",
    "DenoisingDiffusionModel",
    "create_noisy_image",
    # MRI Reconstruction
    "MRIReconstructor",
    "CompressedSensingMRI",
    "SenseReconstruction",
    "GrappaReconstruction",
    "KspaceDeepReconstruction",
    "VarNet",
    "CoilSensitivityEstimator",
    "RSSReconstruction",
    "create_cartesian_mask",
    "create_radial_mask",
    "create_spiral_mask",
    # Tomography
    "TomographyReconstructor",
    "FilteredBackProjection",
    "ART",
    "SIRT",
    "DeepTomography",
    "LearnedIterativeTomography",
    "MLEM",
    "create_sinogram",
    # Regularization
    "Regularizer",
    "TikhonovRegularization",
    "TotalVariationRegularization",
    "L1Regularization",
    "L0Regularization",
    "ElasticNetRegularization",
    "NuclearNormRegularization",
    "SpectralRegularization",
    "GroupLassoRegularization",
    "LearnedRegularization",
    "DeepPriorRegularization",
    "HessianRegularization",
    "WaveletRegularization",
    "FusedRegularization",
    "DirichletRegularization",
    "HuberRegularization",
    "ProximalOperator",
    "create_regularizer",
    # Solvers
    "InverseProblemSolver",
    "ConjugateGradient",
    "ADMM",
    "PrimalDualHybridGradient",
    "GradientDescent",
    "GaussNewton",
    "ProximalGradientDescent",
    "FISTA",
    "SplitBregman",
    "StochasticGradientDescent",
    "LBFGS",
    "create_solver",
]
