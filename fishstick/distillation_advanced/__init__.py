"""
fishstick Distillation Advanced Module

Advanced knowledge distillation techniques including:
- Vanilla KD, label smoothing, multi-teacher distillation
- Feature-based distillation (FitNet, AT, SP, RKD, OFD)
- Self-distillation methods
"""

from fishstick.distillation_advanced.vanilla import (
    VanillaKnowledgeDistillation,
    LabelSmoothingDistillation,
    MultiTeacherDistillation,
    ProgressiveDistillation,
    DynamicTemperatureKD,
)

from fishstick.distillation_advanced.feature import (
    FitNet,
    AttentionTransfer,
    SimilarityPreserving,
    RelationKnowledgeDistillation,
    OFD,
    CombinedFeatureDistillation,
)

from fishstick.distillation_advanced.self import (
    SelfDistillation,
    BeYourOwnTeacher,
    DataFreeKnowledgeDistillation,
    DeepSelfDistillation,
    SnapshotDistillation,
)

__all__ = [
    "VanillaKnowledgeDistillation",
    "LabelSmoothingDistillation",
    "MultiTeacherDistillation",
    "ProgressiveDistillation",
    "DynamicTemperatureKD",
    "FitNet",
    "AttentionTransfer",
    "SimilarityPreserving",
    "RelationKnowledgeDistillation",
    "OFD",
    "CombinedFeatureDistillation",
    "SelfDistillation",
    "BeYourOwnTeacher",
    "DataFreeKnowledgeDistillation",
    "DeepSelfDistillation",
    "SnapshotDistillation",
]
