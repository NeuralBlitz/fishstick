"""
Experiment Tracking & Logging Extensions

Comprehensive experiment tracking tools including:
- Metrics logging utilities
- Checkpoint management
- Experiment configuration
- Results visualization
- Comparison utilities

Modules:
- metrics_logger: Comprehensive metrics logging
- checkpoint_manager: Model checkpoint management
- experiment_config: Configuration management
- results_visualizer: Visualization utilities
- comparison_utils: Comparison and analysis
"""

from .metrics_logger import (
    MetricTracker,
    MetricValue,
    ScalarLogger,
    HistogramLogger,
    ImageLogger,
    TextLogger,
    CSVLogger,
    JSONLogger,
    MetricsLogger,
    CompositeLogger,
)

from .checkpoint_manager import (
    CheckpointMetadata,
    Checkpoint,
    BestModelTracker,
    CheckpointHistory,
    CheckpointOptimizer,
    LazyCheckpoint,
    CheckpointManager,
)

from .experiment_config import (
    DataConfig,
    ModelConfig,
    TrainingConfig,
    ExperimentConfig,
    ConfigValidator,
    ConfigMerger,
    SweepConfig,
    Hyperparameters,
    ConfigLoader,
)

from .results_visualizer import (
    MetricPlotter,
    HeatmapPlotter,
    ConfusionMatrixPlotter,
    TrainingProgressPlotter,
    ComparisonPlotter,
    DistributionPlotter,
    ImageGridPlotter,
    ResultsVisualizer,
)

from .comparison_utils import (
    ExperimentResult,
    ExperimentComparator,
    MetricComparator,
    RankingCalculator,
    StatisticalTests,
    TableFormatter,
    ReportGenerator,
    ExperimentDatabase,
)

__all__ = [
    "MetricTracker",
    "MetricValue",
    "ScalarLogger",
    "HistogramLogger",
    "ImageLogger",
    "TextLogger",
    "CSVLogger",
    "JSONLogger",
    "MetricsLogger",
    "CompositeLogger",
    "CheckpointMetadata",
    "Checkpoint",
    "BestModelTracker",
    "CheckpointHistory",
    "CheckpointOptimizer",
    "LazyCheckpoint",
    "CheckpointManager",
    "DataConfig",
    "ModelConfig",
    "TrainingConfig",
    "ExperimentConfig",
    "ConfigValidator",
    "ConfigMerger",
    "SweepConfig",
    "Hyperparameters",
    "ConfigLoader",
    "MetricPlotter",
    "HeatmapPlotter",
    "ConfusionMatrixPlotter",
    "TrainingProgressPlotter",
    "ComparisonPlotter",
    "DistributionPlotter",
    "ImageGridPlotter",
    "ResultsVisualizer",
    "ExperimentResult",
    "ExperimentComparator",
    "MetricComparator",
    "RankingCalculator",
    "StatisticalTests",
    "TableFormatter",
    "ReportGenerator",
    "ExperimentDatabase",
]
