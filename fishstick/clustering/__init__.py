"""
Clustering Module for fishstick AI Framework
=============================================

Comprehensive clustering algorithms including:
- K-means variants (standard, mini-batch, bisecting)
- Hierarchical clustering (agglomerative, divisive, BIRCH)
- DBSCAN and density-based methods (DBSCAN, OPTICS, MeanShift)
- Spectral clustering (normalized cuts, ratio cuts, kernel, self-tuning)
- Deep clustering methods (DEC, IDEC, DCN, ClusterGAN, VaDE, JULE)

Author: Agent 47
"""

from .base import (
    ClustererBase,
    ClusterResult,
    DistanceMetric,
    GraphBuilder,
    ClusterValidityIndex,
    compute_distance_matrix,
    initialize_centroids,
    compute_inertia,
    relabel_clusters,
)

from .kmeans import (
    KMeans,
    MiniBatchKMeans,
    BisectingKMeans,
    KMeansPlusPlus,
    create_kmeans,
    create_minibatch_kmeans,
    create_bisecting_kmeans,
)

from .hierarchical import (
    AgglomerativeClustering,
    DivisiveClustering,
    BIRCH,
    compute_linkage_matrix,
    cut_tree,
    fcluster,
    create_agglomerative_clustering,
    create_divisive_clustering,
    create_birch,
)

from .dbscan import (
    DBSCAN,
    OPTICS,
    MeanShift,
    MeanShiftTorch,
    DensityPeakClustering,
    create_dbscan,
    create_optics,
    create_meanshift,
)

from .spectral import (
    SpectralClustering,
    NormalizedCutSpectral,
    RatioCutSpectral,
    KernelSpectralClustering,
    SelfTuningSpectral,
    GraphLaplacian,
    create_spectral_clustering,
    create_normalized_spectral,
)

from .deep_clustering import (
    Encoder,
    Decoder,
    AutoEncoder,
    DEC,
    DECTrainer,
    IDEC,
    DCN,
    ClusterGAN,
    VaDE,
    JULE,
    KMeansModel,
    create_dec,
    create_idec,
    create_dcn,
)

__all__ = [
    # Base classes
    "ClustererBase",
    "ClusterResult",
    "DistanceMetric",
    "GraphBuilder",
    "ClusterValidityIndex",
    "compute_distance_matrix",
    "initialize_centroids",
    "compute_inertia",
    "relabel_clusters",
    # K-means variants
    "KMeans",
    "MiniBatchKMeans",
    "BisectingKMeans",
    "KMeansPlusPlus",
    "create_kmeans",
    "create_minibatch_kmeans",
    "create_bisecting_kmeans",
    # Hierarchical
    "AgglomerativeClustering",
    "DivisiveClustering",
    "BIRCH",
    "compute_linkage_matrix",
    "cut_tree",
    "fcluster",
    "create_agglomerative_clustering",
    "create_divisive_clustering",
    "create_birch",
    # DBSCAN and density-based
    "DBSCAN",
    "OPTICS",
    "MeanShift",
    "MeanShiftTorch",
    "DensityPeakClustering",
    "create_dbscan",
    "create_optics",
    "create_meanshift",
    # Spectral clustering
    "SpectralClustering",
    "NormalizedCutSpectral",
    "RatioCutSpectral",
    "KernelSpectralClustering",
    "SelfTuningSpectral",
    "GraphLaplacian",
    "create_spectral_clustering",
    "create_normalized_spectral",
    # Deep clustering
    "Encoder",
    "Decoder",
    "AutoEncoder",
    "DEC",
    "DECTrainer",
    "IDEC",
    "DCN",
    "ClusterGAN",
    "VaDE",
    "JULE",
    "KMeansModel",
    "create_dec",
    "create_idec",
    "create_dcn",
]
