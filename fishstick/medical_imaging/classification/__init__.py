"""
Disease Classification Frameworks

Backbones, multi-task learning, and MIL for medical image classification.
"""

from fishstick.medical_imaging.classification.backbones import (
    ResNet3DMedical,
    DenseNet3D,
    MedicalBackbone,
)

from fishstick.medical_imaging.classification.frameworks import (
    MultiTaskClassifier,
    EnsembleClassifier,
    DiseaseClassifier,
)

from fishstick.medical_imaging.classification.mil import (
    MILPool,
    AttentionMIL,
)

__all__ = [
    "ResNet3DMedical",
    "DenseNet3D",
    "MedicalBackbone",
    "MultiTaskClassifier",
    "EnsembleClassifier",
    "DiseaseClassifier",
    "MILPool",
    "AttentionMIL",
]
