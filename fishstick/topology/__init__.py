"""
Topological Data Analysis (TDA) Module for fishstick.

This module provides tools for:
- Persistent homology computations
- Vietoris-Rips complex construction
- Mapper algorithm implementations
- Topological feature extraction
- TDA-based loss functions for geometric deep learning
"""

from .persistence import (
    PersistentHomology,
    PersistenceDiagram,
    BirthDeathPair,
)
from .vietoris_rips import VietorisRipsComplex, Filtration
from .mapper import Mapper, MapperCover, SimplicialComplexBuilder
from .features import (
    TopologicalFeatures,
    PersistentEntropy,
    BettiCurve,
    PersistenceLandscape,
    Silhouette,
)
from .losses import (
    PersistentHomologyLoss,
    DiagramDistanceLoss,
    PersistentEntropyLoss,
    TopologicalRegularization,
)
from .simplicial import (
    SimplicialComplex,
    BoundaryOperator,
    HomologyBasis,
)

__all__ = [
    "PersistentHomology",
    "PersistenceDiagram",
    "BirthDeathPair",
    "VietorisRipsComplex",
    "Filtration",
    "Mapper",
    "MapperCover",
    "SimplicialComplexBuilder",
    "TopologicalFeatures",
    "PersistentEntropy",
    "BettiCurve",
    "PersistenceLandscape",
    "Silhouette",
    "PersistentHomologyLoss",
    "DiagramDistanceLoss",
    "PersistentEntropyLoss",
    "TopologicalRegularization",
    "SimplicialComplex",
    "BoundaryOperator",
    "HomologyBasis",
]
