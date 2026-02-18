"""
Fishstick Anomaly Detection Module.

A comprehensive collection of anomaly detection algorithms including:
- Reconstruction-based methods
- Distance-based methods
- Density-based methods
- Anomaly scoring functions
- Evaluation metrics
- Visualization tools
"""

from .detection import (
    # Base classes
    BaseAnomalyDetector,
    BaseTorchAnomalyDetector,
    # Reconstruction-based
    AutoencoderAnomalyDetector,
    VAEAnomalyDetector,
    DeepSVDD,
    AnoGAN,
    MemAE,
    # Distance-based
    KNNAnomalyDetector,
    LOFDetector,
    IsolationForestDetector,
    OCSVMDetector,
    # Density-based
    GMMDetector,
    KernelDensityEstimator,
    NormalizingFlowDetector,
    RealNVPAnomaly,
    AffineCouplingLayer,
    # Scoring functions
    reconstruction_error,
    mahalanobis_distance,
    isolation_score,
    energy_score,
    entropy_score,
    # Evaluation metrics
    precision_at_k,
    recall_at_k,
    f1_at_threshold,
    # Visualization
    plot_anomaly_scores,
    plot_roc_curve,
    plot_precision_recall,
    highlight_anomalies,
)

__version__ = "1.0.0"
__all__ = [
    "BaseAnomalyDetector",
    "BaseTorchAnomalyDetector",
    "AutoencoderAnomalyDetector",
    "VAEAnomalyDetector",
    "DeepSVDD",
    "AnoGAN",
    "MemAE",
    "KNNAnomalyDetector",
    "LOFDetector",
    "IsolationForestDetector",
    "OCSVMDetector",
    "GMMDetector",
    "KernelDensityEstimator",
    "NormalizingFlowDetector",
    "RealNVPAnomaly",
    "AffineCouplingLayer",
    "reconstruction_error",
    "mahalanobis_distance",
    "isolation_score",
    "energy_score",
    "entropy_score",
    "precision_at_k",
    "recall_at_k",
    "f1_at_threshold",
    "plot_anomaly_scores",
    "plot_roc_curve",
    "plot_precision_recall",
    "highlight_anomalies",
]
