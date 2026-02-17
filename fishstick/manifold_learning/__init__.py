"""
Manifold Learning and Dimensionality Reduction.

Advanced manifold learning algorithms for extracting low-dimensional
structure from high-dimensional data.

Modules:
- base: Core utilities and base classes
- isomap: Isometric Mapping
- lle: Locally Linear Embedding variants
- diffusion_maps: Diffusion Maps
- hessian_eigenmaps: Hessian Eigenmaps
- manifold_regularization: Manifold regularization for neural networks
"""

from .base import (
    ManifoldEmbedding,
    GraphBuilder,
    KernelBuilder,
    EigenSolver,
    ManifoldLearnerBase,
    compute_distance_matrix,
    compute_gaussian_kernel,
    local_pca,
)

from .isomap import (
    Isomap,
    LandmarkIsomap,
    IsomapLayer,
    create_isomap,
)

from .lle import (
    LLE,
    ModifiedLLE,
    HessianLLE,
    LTSA,
    LLELayer,
    create_lle,
)

from .diffusion_maps import (
    DiffusionMap,
    MultiscaleDiffusionMap,
    AnisotropicDiffusionMap,
    KernelPCA,
    DiffusionMapLayer,
    create_diffusion_map,
)

from .hessian_eigenmaps import (
    HessianEigenmaps,
    HessianLLE,
    CurvatureBasedEmbedding,
    HessianEigenmapsLayer,
    create_hessian_eigenmaps,
)

from .manifold_regularization import (
    LaplaceBeltramiOperator,
    ManifoldRegularizationLoss,
    GraphBasedRegularization,
    ManifoldRegularizationLayer,
    IntrinsicDimensionEstimator,
    ManifoldMetric,
    ManifoldSmoothingLayer,
    ManifoldAugmentation,
    create_manifold_regularizer,
    manifold_loss,
)


__all__ = [
    "ManifoldEmbedding",
    "GraphBuilder",
    "KernelBuilder",
    "EigenSolver",
    "ManifoldLearnerBase",
    "compute_distance_matrix",
    "compute_gaussian_kernel",
    "local_pca",
    "Isomap",
    "LandmarkIsomap",
    "IsomapLayer",
    "create_isomap",
    "LLE",
    "ModifiedLLE",
    "HessianLLE",
    "LTSA",
    "LLELayer",
    "create_lle",
    "DiffusionMap",
    "MultiscaleDiffusionMap",
    "AnisotropicDiffusionMap",
    "KernelPCA",
    "DiffusionMapLayer",
    "create_diffusion_map",
    "HessianEigenmaps",
    "CurvatureBasedEmbedding",
    "HessianEigenmapsLayer",
    "create_hessian_eigenmaps",
    "LaplaceBeltramiOperator",
    "ManifoldRegularizationLoss",
    "GraphBasedRegularization",
    "ManifoldRegularizationLayer",
    "IntrinsicDimensionEstimator",
    "ManifoldMetric",
    "ManifoldSmoothingLayer",
    "ManifoldAugmentation",
    "create_manifold_regularizer",
    "manifold_loss",
]
