"""
Advanced Model Compression Tools for fishstick

Comprehensive model compression utilities including:
- Quantization: PTQ, dynamic, static, QAT, mixed-precision
- Pruning: Magnitude, lottery ticket, movement, structured
- Distillation: Multi-teacher, self-distillation, feature-based
- NAS: Search space, supernet, evolution search
- Speedup: Layer fusion, profiling, benchmarking, optimization

References:
- https://arxiv.org/abs/1803.03635 (Lottery Ticket Hypothesis)
- https://arxiv.org/abs/1503.02531 (Distilling Knowledge)
- https://arxiv.org/abs/1909.10836 (Once for All)
- https://pytorch.org/docs/stable/quantization.html
"""

from __future__ import annotations

from typing import Optional, Dict, Any, Union, List, Tuple
import torch
from torch import nn

try:
    from .quantization_advanced import (
        PTQQuantizer,
        DynamicQuantizationEngine,
        StaticQuantizationEngine,
        MixedPrecisionQuantizer,
        QuantizationAwareTrainer,
        FakeQuantize,
        QuantizationObserver,
        QuantizationMode,
        CalibrationMethod,
    )

    _QUANTIZATION_AVAILABLE = True
except ImportError as e:
    _QUANTIZATION_AVAILABLE = False
    _QUANTIZATION_ERROR = str(e)

try:
    from .pruning_advanced import (
        MagnitudePrunerAdvanced,
        LotteryTicketFinder,
        MovementPruner,
        StructuredPrunerAdvanced,
        PruningScheduler,
        SensitivityPruner,
        PruningSchedule,
        PruningType,
    )

    _PRUNING_AVAILABLE = True
except ImportError as e:
    _PRUNING_AVAILABLE = False
    _PRUNING_ERROR = str(e)

try:
    from .distillation_advanced import (
        KnowledgeDistiller,
        MultiTeacherDistiller,
        SelfDistiller,
        FeatureRepresentationDistiller,
        AdaptiveDistillationLoss,
        AttentionTransfer,
        DistillationType,
    )

    _DISTILLATION_AVAILABLE = True
except ImportError as e:
    _DISTILLATION_AVAILABLE = False
    _DISTILLATION_ERROR = str(e)

try:
    from .nas_primitives import (
        SearchSpace,
        SuperNet,
        ArchitectureSampler,
        PerformanceEstimator,
        EvolutionSearch,
        LatencyAwareSearch,
        SearchSpaceType,
        LayerChoice,
    )

    _NAS_AVAILABLE = True
except ImportError as e:
    _NAS_AVAILABLE = False
    _NAS_ERROR = str(e)

try:
    from .speedup_utils import (
        ModelSpeedupProfiler,
        LayerFuser,
        InferenceOptimizer,
        ModelBenchmarker,
        MemoryEfficientForward,
        OperatorFusion,
        JITOptimizer,
        ActivationCheckpointing,
    )

    _SPEEDUP_AVAILABLE = True
except ImportError as e:
    _SPEEDUP_AVAILABLE = False
    _SPEEDUP_ERROR = str(e)


def get_compression_pipeline(
    model: nn.Module,
    quantization: bool = True,
    pruning: bool = False,
    distillation_teacher: Optional[nn.Module] = None,
) -> nn.Module:
    """Create a full compression pipeline.

    Args:
        model: Model to compress
        quantization: Whether to apply quantization
        pruning: Whether to apply pruning
        distillation_teacher: Optional teacher for distillation

    Returns:
        Compressed model
    """
    compressed_model = model

    if distillation_teacher is not None and _DISTILLATION_AVAILABLE:
        distiller = KnowledgeDistiller(
            compressed_model,
            distillation_teacher,
            temperature=4.0,
            alpha=0.7,
        )
        compressed_model = distiller.student

    if pruning and _PRUNING_AVAILABLE:
        pruner = MagnitudePrunerAdvanced(
            compressed_model,
            initial_sparsity=0.0,
            final_sparsity=0.5,
        )
        compressed_model = pruner.model

    if quantization and _QUANTIZATION_AVAILABLE:
        quantizer = DynamicQuantizationEngine(compressed_model)
        compressed_model = quantizer.quantize()

    return compressed_model


def estimate_compression_ratio(
    original_model: nn.Module,
    compressed_model: nn.Module,
) -> Dict[str, float]:
    """Estimate compression ratio after compression.

    Args:
        original_model: Original uncompressed model
        compressed_model: Compressed model

    Returns:
        Dict with compression metrics
    """
    original_params = sum(p.numel() for p in original_model.parameters())
    compressed_params = sum(p.numel() for p in compressed_model.parameters())

    original_size = sum(
        p.numel() * p.element_size() for p in original_model.parameters()
    )
    compressed_size = sum(
        p.numel() * p.element_size() for p in compressed_model.parameters()
    )

    return {
        "parameter_ratio": original_params / max(compressed_params, 1),
        "size_ratio": original_size / max(compressed_size, 1),
        "original_params": original_params,
        "compressed_params": compressed_params,
    }


__all__ = [
    "PTQQuantizer",
    "DynamicQuantizationEngine",
    "StaticQuantizationEngine",
    "MixedPrecisionQuantizer",
    "QuantizationAwareTrainer",
    "FakeQuantize",
    "QuantizationObserver",
    "QuantizationMode",
    "CalibrationMethod",
    "MagnitudePrunerAdvanced",
    "LotteryTicketFinder",
    "MovementPruner",
    "StructuredPrunerAdvanced",
    "PruningScheduler",
    "SensitivityPruner",
    "PruningSchedule",
    "PruningType",
    "KnowledgeDistiller",
    "MultiTeacherDistiller",
    "SelfDistiller",
    "FeatureRepresentationDistiller",
    "AdaptiveDistillationLoss",
    "AttentionTransfer",
    "DistillationType",
    "SearchSpace",
    "SuperNet",
    "ArchitectureSampler",
    "PerformanceEstimator",
    "EvolutionSearch",
    "LatencyAwareSearch",
    "SearchSpaceType",
    "LayerChoice",
    "ModelSpeedupProfiler",
    "LayerFuser",
    "InferenceOptimizer",
    "ModelBenchmarker",
    "MemoryEfficientForward",
    "OperatorFusion",
    "JITOptimizer",
    "ActivationCheckpointing",
    "get_compression_pipeline",
    "estimate_compression_ratio",
]

__version__ = "0.1.0"
