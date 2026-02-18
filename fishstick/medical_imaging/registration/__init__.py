"""
Medical Image Registration Module

Affine, rigid, and deformable registration for medical images.
"""

from fishstick.medical_imaging.registration.affine import (
    AffineRegistration,
    RigidRegistration,
)

from fishstick.medical_imaging.registration.deformable import (
    DeformableRegistration,
    DemonsRegistration,
)

from fishstick.medical_imaging.registration.transforms import (
    RegistrationResult,
    apply_transform,
    compose_transforms,
    compute_jacobian_determinant,
    SymmetricNormalization,
    TransformationField,
)

__all__ = [
    "AffineRegistration",
    "RigidRegistration",
    "DeformableRegistration",
    "DemonsRegistration",
    "RegistrationResult",
    "apply_transform",
    "compose_transforms",
    "compute_jacobian_determinant",
    "SymmetricNormalization",
    "TransformationField",
]
