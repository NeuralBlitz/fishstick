"""
Topological Data Analysis (TDA) Module for fishstick.

This module provides tools for:
- Persistent homology computations
- Vietoris-Rips complex construction
- Mapper algorithm implementations
- Topological feature extraction
- TDA-based loss functions for geometric deep learning
- Cohomology and alternative complex builders
- Topological kernels and signatures
- Statistical inference for TDA
- Multi-scale analysis
"""

from .persistence import (
    PersistentHomology,
    PersistenceDiagram,
    BirthDeathPair,
)
from .vietoris_rips import VietorisRipsComplex, Filtration
from .mapper import Mapper, MapperCover, SimplicialComplexBuilder
from .features import (
    TopologicalFeatures,
    PersistentEntropy,
    BettiCurve,
    PersistenceLandscape,
    Silhouette,
)
from .losses import (
    PersistentHomologyLoss,
    DiagramDistanceLoss,
    PersistentEntropyLoss,
    TopologicalRegularization,
)
from .simplicial import (
    SimplicialComplex,
    BoundaryOperator,
    HomologyBasis,
)
from .cohomology import (
    PersistentCohomology,
    CohomologyClass,
    SteenrodAlgebra,
    DualizedPersistence,
    riemann_ross_persistence_integral,
)
from .complexes import (
    CechComplex,
    WitnessComplex,
    LazyWitnessComplex,
    GridComplex,
    NerveComplex,
)
from .barcode import (
    PersistenceBarcode,
    BarcodeInterval,
    BarcodeAnalyzer,
    BarcodeComparator,
    diagram_to_barcode,
)
from .kernels import (
    PersistenceScaleSpaceKernel,
    PersistenceHeatKernel,
    WeightedKernel,
    SlicedWassersteinKernel,
    PersistenceFisherKernel,
    TopologicalKernelMatrix,
)
from .signatures import (
    PersistenceImage,
    TopologicalVectorization,
    CombinedSignature,
    MultiScaleSignature,
    AdaptiveSignature,
    compute_all_signatures,
)
from .topo_layers import (
    PersistentHomologyLayer,
    PersistencePooling,
    TopologicalAttention,
    TopologicalMessagePassing,
    PersistentGraphConv,
    TopologicalPool,
    FiltrationAwareEmbedding,
    TopologicalAggregation,
)
from .geometric_losses import (
    GraphTopologicalRegularization,
    PersistentGraphAlignmentLoss,
    TopologicalGraphDistillationLoss,
    HomologicalConnectivityLoss,
    SimplicialNeuralNetworkLoss,
    TopologicalGraphMatchingLoss,
    BoundaryOperatorLoss,
    LaplacianSmoothingLoss,
)
from .aggregation import (
    PersistenceAggregator,
    TopologicalStatistics,
    MultiScaleAggregator,
    BatchAggregator,
    AttentionAggregator,
    KernelAggregator,
    aggregate_topological_features,
)
from .statistics import (
    BootstrapConfidenceInterval,
    HypothesisTest,
    PersistenceDistributionFitting,
    StabilityAnalyzer,
    PersistenceFeatureExtractor,
    confidence_interval_bootstrap,
    permutation_test,
)
from .multiscale import (
    MultiScaleFiltration,
    AdaptiveScaleSelection,
    HierarchicalPersistence,
    ScaleSelection,
    ScaleSpaceEmbedding,
    compute_optimal_filtration_scale,
    multi_scale_persistence_features,
)

__all__ = [
    "PersistentHomology",
    "PersistenceDiagram",
    "BirthDeathPair",
    "VietorisRipsComplex",
    "Filtration",
    "Mapper",
    "MapperCover",
    "SimplicialComplexBuilder",
    "TopologicalFeatures",
    "PersistentEntropy",
    "BettiCurve",
    "PersistenceLandscape",
    "Silhouette",
    "PersistentHomologyLoss",
    "DiagramDistanceLoss",
    "PersistentEntropyLoss",
    "TopologicalRegularization",
    "SimplicialComplex",
    "BoundaryOperator",
    "HomologyBasis",
    "PersistentCohomology",
    "CohomologyClass",
    "SteenrodAlgebra",
    "DualizedPersistence",
    "riemann_ross_persistence_integral",
    "CechComplex",
    "WitnessComplex",
    "LazyWitnessComplex",
    "GridComplex",
    "NerveComplex",
    "PersistenceBarcode",
    "BarcodeInterval",
    "BarcodeAnalyzer",
    "BarcodeComparator",
    "diagram_to_barcode",
    "PersistenceScaleSpaceKernel",
    "PersistenceHeatKernel",
    "WeightedKernel",
    "SlicedWassersteinKernel",
    "PersistenceFisherKernel",
    "TopologicalKernelMatrix",
    "PersistenceImage",
    "TopologicalVectorization",
    "CombinedSignature",
    "MultiScaleSignature",
    "AdaptiveSignature",
    "compute_all_signatures",
    "PersistentHomologyLayer",
    "PersistencePooling",
    "TopologicalAttention",
    "TopologicalMessagePassing",
    "PersistentGraphConv",
    "TopologicalPool",
    "FiltrationAwareEmbedding",
    "TopologicalAggregation",
    "GraphTopologicalRegularization",
    "PersistentGraphAlignmentLoss",
    "TopologicalGraphDistillationLoss",
    "HomologicalConnectivityLoss",
    "SimplicialNeuralNetworkLoss",
    "TopologicalGraphMatchingLoss",
    "BoundaryOperatorLoss",
    "LaplacianSmoothingLoss",
    "PersistenceAggregator",
    "TopologicalStatistics",
    "MultiScaleAggregator",
    "BatchAggregator",
    "AttentionAggregator",
    "KernelAggregator",
    "aggregate_topological_features",
    "BootstrapConfidenceInterval",
    "HypothesisTest",
    "PersistenceDistributionFitting",
    "StabilityAnalyzer",
    "PersistenceFeatureExtractor",
    "confidence_interval_bootstrap",
    "permutation_test",
    "MultiScaleFiltration",
    "AdaptiveScaleSelection",
    "HierarchicalPersistence",
    "ScaleSelection",
    "ScaleSpaceEmbedding",
    "compute_optimal_filtration_scale",
    "multi_scale_persistence_features",
]
