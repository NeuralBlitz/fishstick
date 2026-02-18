"""
Bioinformatics Module for fishstick AI Framework.

This module provides primitives for computational biology and bioinformatics tasks:
- Protein structure prediction
- DNA/RNA sequence analysis
- Gene expression modeling
- Single-cell analysis
- Drug discovery primitives

All modules use PyTorch for tensor operations and follow the fishstick
mathematical rigor and type safety conventions.
"""

from typing import TYPE_CHECKING

try:
    from .protein import (
        ProteinSequence,
        ProteinEncoder,
        SecondaryStructurePredictor,
        TorsionAngles,
        ContactMapPredictor,
    )

    _PROTEIN_AVAILABLE = True
except ImportError:
    _PROTEIN_AVAILABLE = False

try:
    from .sequence_analysis import (
        NucleotideEncoder,
        KMerCounter,
        MotifFinder,
        PromoterPredictor,
        VariantEffectPredictor,
    )

    _SEQUENCE_AVAILABLE = True
except ImportError:
    _SEQUENCE_AVAILABLE = False

try:
    from .expression import (
        ExpressionNormalizer,
        TPMNormalizer,
        FPKMNormalizer,
        DifferentialExpression,
        PathwayEnrichment,
    )

    _EXPRESSION_AVAILABLE = True
except ImportError:
    _EXPRESSION_AVAILABLE = False

try:
    from .singlecell import (
        ScNormalizer,
        ScTransform,
        SingleCellClustering,
        TrajectoryInference,
        BatchCorrector,
    )

    _SINGLECELL_AVAILABLE = True
except ImportError:
    _SINGLECELL_AVAILABLE = False

try:
    from .drug_discovery import (
        MoleculeEncoder,
        MolecularFingerprint,
        BindingAffinityPredictor,
        MoleculeGenerator,
        DrugLikenessFilter,
    )

    _DRUG_AVAILABLE = True
except ImportError:
    _DRUG_AVAILABLE = False

__all__ = [
    "ProteinSequence",
    "ProteinEncoder",
    "SecondaryStructurePredictor",
    "TorsionAngles",
    "ContactMapPredictor",
    "NucleotideEncoder",
    "KMerCounter",
    "MotifFinder",
    "PromoterPredictor",
    "VariantEffectPredictor",
    "ExpressionNormalizer",
    "TPMNormalizer",
    "FPKMNormalizer",
    "DifferentialExpression",
    "PathwayEnrichment",
    "ScNormalizer",
    "ScTransform",
    "SingleCellClustering",
    "TrajectoryInference",
    "BatchCorrector",
    "MoleculeEncoder",
    "MolecularFingerprint",
    "BindingAffinityPredictor",
    "MoleculeGenerator",
    "DrugLikenessFilter",
]
