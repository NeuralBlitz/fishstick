"""
fishstick Knowledge Distillation Module

Knowledge distillation utilities and algorithms.
"""

from fishstick.distillation.base import DistillationLoss, TemperatureScaledLoss
from fishstick.distillation.behaviors import FeatureDistillation, RelationDistillation
from fishstick.distillation.advanced import (
    KnowledgeDistillationLoss,
    FeatureDistillationLoss,
    TakeKD,
    DeepMutualLearning,
    AttentionTransfer,
)

__all__ = [
    "DistillationLoss",
    "TemperatureScaledLoss",
    "FeatureDistillation",
    "RelationDistillation",
    "KnowledgeDistillationLoss",
    "FeatureDistillationLoss",
    "TakeKD",
    "DeepMutualLearning",
    "AttentionTransfer",
]
