"""
fishstick
=========
A mathematically rigorous, physically grounded AI framework synthesizing:
- Theoretical Physics (symmetry, renormalization, variational principles)
- Formal Mathematics (category theory, sheaf cohomology, type theory)
- Advanced Machine Learning (equivariant DL, neuro-symbolic, thermodynamic bounds)

This package implements 6 unified frameworks:
- UniIntelli (A.md): Categorical–Geometric–Thermodynamic Synthesis
- HSCA (B.md): Holo-Symplectic Cognitive Architecture
- UIA (C.md): Unified Intelligence Architecture
- SCIF (D.md): Symplectic-Categorical Intelligence Framework
- UIF (E.md): Unified Intelligence Framework
- UIS (F.md): Unified Intelligence Synthesis
"""

__version__ = "0.1.0"
__author__ = "NeuralBlitz"

from .core.types import (
    MetricTensor,
    SymplecticForm,
    Connection,
    ProbabilisticState,
    PhaseSpaceState,
    ConservationLaw,
    VerificationCertificate,
)
from .core.manifold import StatisticalManifold
from .categorical.category import (
    MonoidalCategory,
    Functor,
    NaturalTransformation,
    TracedMonoidalCategory,
    DaggerCategory,
)
from .categorical.lens import Lens, BidirectionalLearner
from .geometric.fisher import FisherInformationMetric, NaturalGradient
from .geometric.sheaf import DataSheaf, SheafCohomology
from .dynamics.hamiltonian import HamiltonianNeuralNetwork, SymplecticIntegrator
from .dynamics.thermodynamic import ThermodynamicGradientFlow, FreeEnergy
from .sheaf.attention import SheafOptimizedAttention
from .rg.autoencoder import RGAutoencoder, RGFlow
from .verification.types import DependentlyTypedLearner, VerificationCertificate

# Import advanced features
try:
    from .neural_ode import (
        ODEFunction,
        NeuralODE,
        AugmentedNeuralODE,
        LatentODE,
        SecondOrderNeuralODE,
        ContinuousNormalizingFlow,
    )

    _NEURAL_ODE_AVAILABLE = True
except ImportError:
    _NEURAL_ODE_AVAILABLE = False

try:
    from .graph import (
        EquivariantMessagePassing,
        SheafGraphConv,
        GeometricGraphTransformer,
        MolecularGraphNetwork,
    )

    _GRAPH_AVAILABLE = True
except ImportError:
    _GRAPH_AVAILABLE = False

try:
    from .probabilistic import (
        BayesianLinear,
        MCDropout,
        VariationalLayer,
        DeepEnsemble,
        EvidentialLayer,
        BayesianNeuralNetwork,
    )

    _PROBABILISTIC_AVAILABLE = True
except ImportError:
    _PROBABILISTIC_AVAILABLE = False

try:
    from .flows import (
        RealNVP,
        Glow,
        MAF,
        ConditionalNormalizingFlow,
    )

    _FLOWS_AVAILABLE = True
except ImportError:
    _FLOWS_AVAILABLE = False

try:
    from .equivariant import (
        SE3EquivariantLayer,
        SE3Transformer,
        E3Conv,
        EquivariantPointCloudNetwork,
    )

    _EQUIVARIANT_AVAILABLE = True
except ImportError:
    _EQUIVARIANT_AVAILABLE = False

try:
    from .causal import (
        CausalGraph,
        StructuralEquation,
        StructuralCausalModel,
        CausalDiscovery,
        CausalVAE,
    )

    _CAUSAL_AVAILABLE = True
except ImportError:
    _CAUSAL_AVAILABLE = False

try:
    from .quantum import (
        QuantumCircuit,
        Gate,
        Hadamard,
        CNOT,
        RX,
        RY,
        RZ,
        PauliX,
        PauliY,
        PauliZ,
        TensorNetwork,
        MPS,
        TTN,
        PEPS,
        QuantumEmbedding,
        AmplitudeEmbedding,
        AngleEmbedding,
        BasicEntanglerLayers,
        QuantumConv1D,
        QuantumConv2D,
    )

    _QUANTUM_AVAILABLE = True
except ImportError:
    _QUANTUM_AVAILABLE = False

try:
    from .relativity import (
        LorentzTransformation,
        Boost,
        Rotation,
        FourVector,
        MinkowskiMetric,
        ProperTime,
        RelativisticParticle,
        EnergyMomentum,
        GeodesicEquation,
        SchwarzschildMetric,
        KerrMetric,
        SpacetimeInterval,
        LightCone,
        Causality,
    )

    _RELATIVITY_AVAILABLE = True
except ImportError:
    _RELATIVITY_AVAILABLE = False

try:
    from .algtopo import (
        SimplicialComplex,
        Simplex,
        Chain,
        Boundary,
        HomologyGroup,
        BettiNumbers,
        Cocycle,
        Coboundary,
        CohomologyGroup,
        CupProduct,
        DeRhamComplex,
        VietorisRipsComplex,
        filtration,
        persistence_diagram,
        bottleneck_distance,
        wasserstein_distance,
        PersistentHomology,
    )

    _ALGTOPO_AVAILABLE = True
except ImportError:
    _ALGTOPO_AVAILABLE = False

try:
    from .representation import (
        LieAlgebra,
        LieGroup,
        su2,
        so3,
        sl2c,
        StructureConstants,
        GroupRepresentation,
        IrreducibleRepresentation,
        TensorRepresentation,
        Character,
        DirectSum,
        TensorProduct,
        WeylGroup,
        RootSystem,
        WeightLattice,
        DynkinDiagram,
    )

    _REPRESENTATION_AVAILABLE = True
except ImportError:
    _REPRESENTATION_AVAILABLE = False


__all__ = [
    "MetricTensor",
    "SymplecticForm",
    "Connection",
    "ProbabilisticState",
    "PhaseSpaceState",
    "ConservationLaw",
    "VerificationCertificate",
    "StatisticalManifold",
    "MonoidalCategory",
    "Functor",
    "NaturalTransformation",
    "TracedMonoidalCategory",
    "DaggerCategory",
    "Lens",
    "BidirectionalLearner",
    "FisherInformationMetric",
    "NaturalGradient",
    "DataSheaf",
    "SheafCohomology",
    "HamiltonianNeuralNetwork",
    "SymplecticIntegrator",
    "ThermodynamicGradientFlow",
    "FreeEnergy",
    "SheafOptimizedAttention",
    "RGAutoencoder",
    "RGFlow",
    "DependentlyTypedLearner",
    "VerificationCertificate",
    # Neural ODE
    "ODEFunction",
    "NeuralODE",
    "AugmentedNeuralODE",
    "LatentODE",
    # Graph
    "EquivariantMessagePassing",
    "SheafGraphConv",
    "GeometricGraphTransformer",
    # Probabilistic
    "BayesianLinear",
    "MCDropout",
    "VariationalLayer",
    "DeepEnsemble",
    "BayesianNeuralNetwork",
    # Flows
    "RealNVP",
    "Glow",
    "MAF",
    # Equivariant
    "SE3EquivariantLayer",
    "SE3Transformer",
    "E3Conv",
    "EquivariantPointCloudNetwork",
    # Causal
    "CausalGraph",
    "StructuralCausalModel",
    "CausalDiscovery",
    # Quantum
    "QuantumCircuit",
    "Gate",
    "Hadamard",
    "CNOT",
    "RX",
    "RY",
    "RZ",
    "PauliX",
    "PauliY",
    "PauliZ",
    "TensorNetwork",
    "MPS",
    "TTN",
    "PEPS",
    "QuantumEmbedding",
    "AmplitudeEmbedding",
    "AngleEmbedding",
    "BasicEntanglerLayers",
    "QuantumConv1D",
    "QuantumConv2D",
    # Relativity
    "LorentzTransformation",
    "Boost",
    "Rotation",
    "FourVector",
    "MinkowskiMetric",
    "ProperTime",
    "RelativisticParticle",
    "EnergyMomentum",
    "GeodesicEquation",
    "SchwarzschildMetric",
    "KerrMetric",
    "SpacetimeInterval",
    "LightCone",
    "Causality",
    # Algebraic Topology
    "SimplicialComplex",
    "Simplex",
    "Chain",
    "Boundary",
    "HomologyGroup",
    "BettiNumbers",
    "Cocycle",
    "Coboundary",
    "CohomologyGroup",
    "CupProduct",
    "DeRhamComplex",
    "VietorisRipsComplex",
    "filtration",
    "persistence_diagram",
    "bottleneck_distance",
    "wasserstein_distance",
    "PersistentHomology",
    # Representation Theory
    "LieAlgebra",
    "LieGroup",
    "su2",
    "so3",
    "sl2c",
    "StructureConstants",
    "GroupRepresentation",
    "IrreducibleRepresentation",
    "TensorRepresentation",
    "Character",
    "DirectSum",
    "TensorProduct",
    "WeylGroup",
    "RootSystem",
    "WeightLattice",
    "DynkinDiagram",
]
