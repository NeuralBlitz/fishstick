"""
fishstick Data Processing Pipeline Module

Comprehensive data processing tools for the fishstick AI framework.

Modules:
- loaders: Advanced data loaders and datasets
- transforms: Data transformation pipelines
- features: Feature engineering utilities
- validation: Data validation tools
- streaming: Streaming data handling

Features:
- Memory-efficient lazy loading
- Composable transformation pipelines
- Feature engineering utilities
- Comprehensive data validation
- Infinite streaming data handling
"""

from __future__ import annotations

from typing import (
    Optional,
    Callable,
    List,
    Union,
    Dict,
    Any,
    Tuple,
)

# Re-export all public APIs
from .loaders import (
    LazyDataset,
    MappedDataset,
    ConcatDataset,
    ChainDataset,
    ShuffleDataset,
    StatefulDataLoader,
    TensorDataset,
    MemoryMappedDataset,
    create_lazy_dataloader,
    create_streaming_dataloader,
)

from .transforms import (
    TransformPipeline,
    TransformStep,
    ConditionalTransform,
    BatchTransform,
    LazyTransform,
    TransformValidator,
    ChainedTransform,
    AdaptiveTransform,
    Compose,
    OneOf,
    RandomApply,
    LambdaTransform,
    IdentityTransform,
    NormalizeTransform,
    DenormalizeTransform,
    ResizeTransform,
    FlattenTransform,
    ToTensorTransform,
    ToNumpyTransform,
    create_transform_pipeline,
)

from .features import (
    PolynomialFeatures,
    InteractionFeatures,
    BinningTransformer,
    TargetEncoder,
    FeatureSelector,
    PCAFeatures,
    FourierFeatures,
    FeatureStatistics,
    compute_feature_statistics,
)

from .validation import (
    ValidationLevel,
    ValidationIssue,
    ValidationReport,
    SchemaValidator,
    RangeValidator,
    StatisticalValidator,
    DuplicateValidator,
    ValidatedDataset,
    DataIntegrityChecker,
    validate_batch,
    validate_dataset,
)

from .streaming import (
    StreamState,
    StreamStats,
    StreamDataLoader,
    BufferedIterator,
    RateLimitedStream,
    CheckpointedStream,
    TransformStream,
    ChunkedStream,
    BatchStream,
    WindowedStream,
    MergedStream,
    AsyncStream,
    create_stream_from_files,
    stream_batches,
)


__all__ = [
    # Loaders
    "LazyDataset",
    "MappedDataset",
    "ConcatDataset",
    "ChainDataset",
    "ShuffleDataset",
    "StatefulDataLoader",
    "TensorDataset",
    "MemoryMappedDataset",
    "create_lazy_dataloader",
    "create_streaming_dataloader",
    # Transforms
    "TransformPipeline",
    "TransformStep",
    "ConditionalTransform",
    "BatchTransform",
    "LazyTransform",
    "TransformValidator",
    "ChainedTransform",
    "AdaptiveTransform",
    "Compose",
    "OneOf",
    "RandomApply",
    "LambdaTransform",
    "IdentityTransform",
    "NormalizeTransform",
    "DenormalizeTransform",
    "ResizeTransform",
    "FlattenTransform",
    "ToTensorTransform",
    "ToNumpyTransform",
    "create_transform_pipeline",
    # Features
    "PolynomialFeatures",
    "InteractionFeatures",
    "BinningTransformer",
    "TargetEncoder",
    "FeatureSelector",
    "PCAFeatures",
    "FourierFeatures",
    "FeatureStatistics",
    "compute_feature_statistics",
    # Validation
    "ValidationLevel",
    "ValidationIssue",
    "ValidationReport",
    "SchemaValidator",
    "RangeValidator",
    "StatisticalValidator",
    "DuplicateValidator",
    "ValidatedDataset",
    "DataIntegrityChecker",
    "validate_batch",
    "validate_dataset",
    # Streaming
    "StreamState",
    "StreamStats",
    "StreamDataLoader",
    "BufferedIterator",
    "RateLimitedStream",
    "CheckpointedStream",
    "TransformStream",
    "ChunkedStream",
    "BatchStream",
    "WindowedStream",
    "MergedStream",
    "AsyncStream",
    "create_stream_from_files",
    "stream_batches",
]


__version__ = "0.1.0"
