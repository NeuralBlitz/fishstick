"""
Fishstick Model Serving Module

Comprehensive model serving infrastructure for deploying ML models at scale.
Supports multiple export formats, serving backends, optimization techniques,
and monitoring capabilities.

Example usage:
    >>> from fishstick.serving import TorchScriptExporter, ModelConfig
    >>> config = ModelConfig("my_model", batch_size=32)
    >>> exporter = TorchScriptExporter(config)
    >>> exporter.export(model, "model.pt")
"""

from .deployment import (
    # Enums
    ExportFormat,
    ServingBackend,
    QuantizationMode,
    # Data classes
    ModelConfig,
    InferenceMetrics,
    HealthStatus,
    # Exporters
    BaseExporter,
    TorchScriptExporter,
    ONNXExporter,
    TensorRTExporter,
    CoreMLExporter,
    OpenVINOExporter,
    # Serving Backends
    BaseServingBackend,
    TorchServeWrapper,
    TritonInferenceServer,
    FastAPIServer,
    FlaskServer,
    gRPCServer,
    # Optimization
    ModelQuantizer,
    ModelPruner,
    BatchOptimizer,
    GraphOptimizer,
    # Request Handling
    RequestPreprocessor,
    RequestValidator,
    RequestBatcher,
    ResponsePostprocessor,
    # Monitoring
    LatencyMonitor,
    ThroughputMonitor,
    MemoryMonitor,
    ErrorTracker,
    # Scaling
    LoadBalancer,
    AutoScaler,
    ModelSharding,
    PipelineParallel,
    PipelineModel,
    # Utilities
    warmup_model,
    benchmark_serving,
    health_check,
)

__version__ = "0.1.0"

__all__ = [
    # Enums
    "ExportFormat",
    "ServingBackend",
    "QuantizationMode",
    # Data classes
    "ModelConfig",
    "InferenceMetrics",
    "HealthStatus",
    # Exporters
    "BaseExporter",
    "TorchScriptExporter",
    "ONNXExporter",
    "TensorRTExporter",
    "CoreMLExporter",
    "OpenVINOExporter",
    # Serving Backends
    "BaseServingBackend",
    "TorchServeWrapper",
    "TritonInferenceServer",
    "FastAPIServer",
    "FlaskServer",
    "gRPCServer",
    # Optimization
    "ModelQuantizer",
    "ModelPruner",
    "BatchOptimizer",
    "GraphOptimizer",
    # Request Handling
    "RequestPreprocessor",
    "RequestValidator",
    "RequestBatcher",
    "ResponsePostprocessor",
    # Monitoring
    "LatencyMonitor",
    "ThroughputMonitor",
    "MemoryMonitor",
    "ErrorTracker",
    # Scaling
    "LoadBalancer",
    "AutoScaler",
    "ModelSharding",
    "PipelineParallel",
    "PipelineModel",
    # Utilities
    "warmup_model",
    "benchmark_serving",
    "health_check",
]
