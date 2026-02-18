"""
fishstick Model Compression Module

Model compression techniques including:
- Pruning: Structured/unstructured, movement, lottery ticket hypothesis
- Quantization: PTQ, QAT, INT8, FP16, mixed precision
- Distillation: Task-agnostic, architecture-agnostic knowledge distillation
"""

from fishstick.model_compression.pruning import (
    MagnitudePruner,
    GradientPruner,
    MovementPruner,
    LotteryTicketPruner,
    StructuredPruner,
    RandomPruner,
)

from fishstick.model_compression.quantization import (
    PTQQuantizer,
    QATQuantizer,
    INT8Quantizer,
    FP16Quantizer,
    MixedPrecisionQuantizer,
    DynamicQuantizer,
    StaticQuantizer,
    FakeQuantizer,
)

from fishstick.model_compression.distillation import (
    TaskAgnosticDistillation,
    ArchitectureAgnosticDistillation,
    ProgressiveDistillation,
    MultiSourceDistillation,
    FeatureBasedDistillation,
    OutputBasedDistillation,
)

__all__ = [
    "MagnitudePruner",
    "GradientPruner",
    "MovementPruner",
    "LotteryTicketPruner",
    "StructuredPruner",
    "RandomPruner",
    "PTQQuantizer",
    "QATQuantizer",
    "INT8Quantizer",
    "FP16Quantizer",
    "MixedPrecisionQuantizer",
    "DynamicQuantizer",
    "StaticQuantizer",
    "FakeQuantizer",
    "TaskAgnosticDistillation",
    "ArchitectureAgnosticDistillation",
    "ProgressiveDistillation",
    "MultiSourceDistillation",
    "FeatureBasedDistillation",
    "OutputBasedDistillation",
]
