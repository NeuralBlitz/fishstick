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
    from .selfsupervised import (
        SimCLR,
        BYOL,
        SimSiam,
        MoCo,
        ContrastiveHead,
        MAE,
        SimMIM,
        MaskedAutoencoder,
        PatchEmbed,
        DeepInfoMax,
        GlobalInfoMax,
        LocalInfoMax,
        BarlowTwins,
        BarlowTwinsLoss,
        NT_XentLoss,
        SimSiamLoss,
        BYOLLoss,
        VicRegLoss,
        InfoNCE,
        BYOLAugmentations,
        SimCLRAugmentations,
        MAEAugmentations,
    )

    _SELFSUPERVISED_AVAILABLE = True
except ImportError:
    _SELFSUPERVISED_AVAILABLE = False

try:
    from .vision import (
        ViT,
        DeiT,
        SwinTransformer,
        CvT,
        create_vit,
        create_deit,
        create_swin,
    )

    _VISION_AVAILABLE = True
except ImportError:
    _VISION_AVAILABLE = False

try:
    from .training.advanced import (
        SAM,
        AdaBelief,
        LAMB,
        WarmupCosineScheduler,
        WarmupLinearScheduler,
        OneCycleLR,
        LabelSmoothingCrossEntropy,
        Mixup,
        CutMix,
    )

    _ADVANCED_TRAINING_AVAILABLE = True
except ImportError:
    _ADVANCED_TRAINING_AVAILABLE = False

try:
    from .distillation.advanced import (
        KnowledgeDistillationLoss,
        FeatureDistillationLoss,
        TakeKD,
        DeepMutualLearning,
        AttentionTransfer,
    )

    _DISTILLATION_AVAILABLE = True
except ImportError:
    _DISTILLATION_AVAILABLE = False

try:
    from .timeseries.forecasting import (
        BaseForecaster,
        LSTMForecaster,
        TransformerForecaster,
        NBeatsForecaster,
        DeepARForecaster,
        FeatureEngineer,
        TimeSeriesMetrics,
        TimeSeriesDataset,
        create_sequences,
        temporal_train_test_split,
        TimeSeriesScaler,
        ScalerType,
        EnsembleForecaster,
        create_forecaster,
        ForecastingTrainer,
    )

    _TIMESERIES_AVAILABLE = True
except ImportError as e:
    print(f"Time series forecasting import error: {e}")
    _TIMESERIES_AVAILABLE = False

try:
    from .selfsupervised.advanced import (
        DINO,
        SwAV,
        WMSE,
        MSN,
    )

    _SSL_ADVANCED_AVAILABLE = True
except ImportError:
    _SSL_ADVANCED_AVAILABLE = False

try:
    from .meta import (
        MAML,
        ProtoNet,
        Reptile,
        MatchingNetwork,
        SNAIL,
    )

    _META_AVAILABLE = True
except ImportError:
    _META_AVAILABLE = False

try:
    from .compression.pruning import (
        MagnitudePruner,
        LotteryTicketPruner,
        DynamicPruner,
        SlimmingPruner,
    )

    _COMPRESSION_AVAILABLE = True
except ImportError:
    _COMPRESSION_AVAILABLE = False

try:
    from .uncertainty import (
        MCAlphaDropout,
        EnsembleUncertainty,
        BayesianNN,
        MaxSoftmaxOODDetector,
        EnergyOODDetector,
        TemperatureScaledClassifier,
    )

    _UNCERTAINTY_AVAILABLE = True
except ImportError:
    _UNCERTAINTY_AVAILABLE = False

try:
    from .continual import (
        EWC,
        PackNet,
        ProgressiveNeuralNetwork,
        MemoryReplay,
        GEM,
        LwF,
    )

    _CONTINUAL_AVAILABLE = True
except ImportError:
    _CONTINUAL_AVAILABLE = False

try:
    from .active.learning import (
        # Query strategies
        UncertaintySampling,
        MarginSampling,
        EntropySampling,
        RandomSampling,
        ClusterBasedSampling,
        DensityWeightedSampling,
        # Uncertainty estimation
        MCDropoutUncertainty,
        EnsembleUncertainty,
        BayesianUncertainty,
        EvidentialUncertainty,
        # Batch active learning
        BatchBALD,
        CoreSet,
        BADGE,
        BatchActive,
        GreedyBatch,
        # Diversity sampling
        KCenterSampling,
        KMeansSampling,
        RepresentativeSampling,
        DiversityAwareSampling,
        AdversarialSampling,
        # Expected model change
        EGL,
        BALD,
        VariationRatio,
        InformationGain,
        # Pool and stream
        PoolBasedAL,
        StreamBasedAL,
        MembershipQuerySynthesis,
        # Multi-task
        MultiTaskAL,
        TransferActive,
        DomainAdaptiveAL,
        # Evaluation
        LearningCurve,
        AnnotationCost,
        ALBenchmark,
        ALVisualization,
        # Integration
        ActiveDataset,
        ActiveTrainer,
        ActiveLoop,
        # Factories
        create_query_strategy,
        create_batch_strategy,
        create_uncertainty_estimator,
        # Utilities
        compute_query_diversity,
        compute_coverage,
        active_learning_summary,
        # Base classes
        QueryStrategy,
        UncertaintyEstimator,
        BatchQueryStrategy,
    )

    _ACTIVE_AVAILABLE = True
except ImportError as e:
    _ACTIVE_AVAILABLE = False

try:
    from .neuroscience import (
        LeakyIntegrateAndFire,
        HodgkinHuxley,
        Izhikevich,
        AdaptiveLIF,
        SpikingDense,
        SpikingConv2d,
        LiquidStateMachine,
        SpikingAttention,
        ThresholdDependentPlasticity,
        STDP,
        OjaRule,
        BCMPlasticity,
        HomeostaticPlasticity,
        TripletSTDP,
        VoltageBasedSTDP,
        NeuralAttention,
        WinnerTakeAllAttention,
        DivisiveNormalization,
        NormalizedAttention,
        FeedbackAttention,
        PredictiveAttention,
        RateEncoder,
        PoissonEncoder,
        TemporalEncoder,
        PopulationEncoder,
        MixedCodeEncoder,
        LatentPopulationEncoder,
        DeltaEncoder,
        GridCellEncoder,
    )

    _NEUROSCIENCE_AVAILABLE = True
except ImportError:
    _NEUROSCIENCE_AVAILABLE = False

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

try:
    from .memory import (
        DifferentiableStack,
        DifferentiableQueue,
        DifferentiableDeque,
        PriorityQueue,
        DifferentiableStackEnsemble,
        NeuralTuringMachine,
        LookupFreeNTM,
        create_ntm,
        HopfieldAttention,
        SparseMemoryAttention,
        KeyValueMemoryAttention,
        AssociativeAttention,
        MemoryAugmentedAttention,
        RoutingAttention,
        SetAttention,
        CosineSimilarityAddressing,
        EuclideanDistanceAddressing,
        DotProductAddressing,
        LearnedSimilarityAddressing,
        MultiHeadContentAddressing,
        HybridAddressing,
        AttentionBasedAddressing,
        MemoryBank,
        AssociativeMemory,
        WorkingMemory,
        EpisodicMemory,
        SemanticMemory,
        HierarchicalMemorySystem,
        ContinualLearningMemory,
        MetaLearningMemory,
    )

    _MEMORY_AVAILABLE = True
except ImportError as e:
    print(f"Memory module import error: {e}")
    _MEMORY_AVAILABLE = False

try:
    from .topology import (
        PersistentHomology,
        PersistenceDiagram,
        BirthDeathPair,
        VietorisRipsComplex,
        Filtration,
        Mapper,
        MapperCover,
        SimplicialComplexBuilder,
        TopologicalFeatures,
        PersistentEntropy,
        BettiCurve,
        PersistenceLandscape,
        Silhouette,
        PersistentHomologyLoss,
        DiagramDistanceLoss,
        PersistentEntropyLoss,
        TopologicalRegularization,
        SimplicialComplex,
        BoundaryOperator,
        HomologyBasis,
    )

    _TOPOLOGY_AVAILABLE = True
except ImportError as e:
    print(f"Topology module import error: {e}")
    _TOPOLOGY_AVAILABLE = False


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
    # Self-Supervised Learning
    "SimCLR",
    "BYOL",
    "SimSiam",
    "MoCo",
    "ContrastiveHead",
    "MAE",
    "SimMIM",
    "MaskedAutoencoder",
    "PatchEmbed",
    "DeepInfoMax",
    "GlobalInfoMax",
    "LocalInfoMax",
    "BarlowTwins",
    "BarlowTwinsLoss",
    "NT_XentLoss",
    "SimSiamLoss",
    "BYOLLoss",
    "VicRegLoss",
    "InfoNCE",
    "BYOLAugmentations",
    "SimCLRAugmentations",
    "MAEAugmentations",
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
    # Vision Transformers
    "ViT",
    "DeiT",
    "SwinTransformer",
    "CvT",
    "create_vit",
    "create_deit",
    "create_swin",
    # Advanced Training
    "SAM",
    "AdaBelief",
    "LAMB",
    "WarmupCosineScheduler",
    "WarmupLinearScheduler",
    "OneCycleLR",
    "LabelSmoothingCrossEntropy",
    "Mixup",
    "CutMix",
    # Knowledge Distillation
    "KnowledgeDistillationLoss",
    "FeatureDistillationLoss",
    "TakeKD",
    "DeepMutualLearning",
    "AttentionTransfer",
    # Time Series Forecasting
    "BaseForecaster",
    "LSTMForecaster",
    "TransformerForecaster",
    "NBeatsForecaster",
    "DeepARForecaster",
    "FeatureEngineer",
    "TimeSeriesMetrics",
    "TimeSeriesDataset",
    "create_sequences",
    "temporal_train_test_split",
    "TimeSeriesScaler",
    "ScalerType",
    "EnsembleForecaster",
    "create_forecaster",
    "ForecastingTrainer",
    # Additional SSL
    "DINO",
    "SwAV",
    "WMSE",
    "MSN",
    # Meta-Learning
    "MAML",
    "ProtoNet",
    "Reptile",
    "MatchingNetwork",
    "SNAIL",
    # Model Compression
    "MagnitudePruner",
    "LotteryTicketPruner",
    "DynamicPruner",
    "SlimmingPruner",
    # Uncertainty
    "MCAlphaDropout",
    "EnsembleUncertainty",
    "BayesianNN",
    "MaxSoftmaxOODDetector",
    "EnergyOODDetector",
    "TemperatureScaledClassifier",
    # Continual Learning
    "EWC",
    "PackNet",
    "ProgressiveNeuralNetwork",
    "MemoryReplay",
    "GEM",
    "LwF",
    # Active Learning
    "UncertaintySampling",
    "MarginSampling",
    "EntropySampling",
    "RandomSampling",
    "ClusterBasedSampling",
    "DensityWeightedSampling",
    "MCDropoutUncertainty",
    "EnsembleUncertainty",
    "BayesianUncertainty",
    "EvidentialUncertainty",
    "BatchBALD",
    "CoreSet",
    "BADGE",
    "BatchActive",
    "GreedyBatch",
    "KCenterSampling",
    "KMeansSampling",
    "RepresentativeSampling",
    "DiversityAwareSampling",
    "AdversarialSampling",
    "EGL",
    "BALD",
    "VariationRatio",
    "InformationGain",
    "PoolBasedAL",
    "StreamBasedAL",
    "MembershipQuerySynthesis",
    "MultiTaskAL",
    "TransferActive",
    "DomainAdaptiveAL",
    "LearningCurve",
    "AnnotationCost",
    "ALBenchmark",
    "ALVisualization",
    "ActiveDataset",
    "ActiveTrainer",
    "ActiveLoop",
    "create_query_strategy",
    "create_batch_strategy",
    "create_uncertainty_estimator",
    "compute_query_diversity",
    "compute_coverage",
    "active_learning_summary",
    "QueryStrategy",
    "UncertaintyEstimator",
    "BatchQueryStrategy",
    # Neuroscience
    "LeakyIntegrateAndFire",
    "HodgkinHuxley",
    "Izhikevich",
    "AdaptiveLIF",
    "SpikingDense",
    "SpikingConv2d",
    "LiquidStateMachine",
    "SpikingAttention",
    "ThresholdDependentPlasticity",
    "STDP",
    "OjaRule",
    "BCMPlasticity",
    "HomeostaticPlasticity",
    "TripletSTDP",
    "VoltageBasedSTDP",
    "NeuralAttention",
    "WinnerTakeAllAttention",
    "DivisiveNormalization",
    "NormalizedAttention",
    "FeedbackAttention",
    "PredictiveAttention",
    "RateEncoder",
    "PoissonEncoder",
    "TemporalEncoder",
    "PopulationEncoder",
    "MixedCodeEncoder",
    "LatentPopulationEncoder",
    "DeltaEncoder",
    "GridCellEncoder",
    # Privacy & Differential Privacy
    "PrivacyEngine",
    "FederatedPrivacyEngine",
    "PrivacyEngineConfig",
    "DPSGD",
    "DPTrainer",
    "GaussianMechanism",
    "LaplaceMechanism",
    "ExponentialMechanism",
    "RDPAccountant",
    "BasicAccountant",
    "GaussianAccountant",
    "StaticClipper",
    "AdaptiveClipler",
    "PerLayerClipper",
    "SubsampleAmplifier",
    "ShuffleAmplifier",
    "PoissonSampler",
    "BatchSampler",
    "NoisyAggregator",
    "DPFederatedAggregator",
    "SecureAggregator",
    "SecureAggregationProtocol",
    "AdditiveSecretSharing",
    "SecureSum",
    "create_privacy_engine",
    "create_noise_mechanism",
    "create_accountant",
    "create_clipper",
    "create_aggregator",
    "create_sampler",
    "amplify_privacy",
    "compose_epsilons",
    "compute_gaussian_epsilon",
    "estimate_epsilon",
    "PrivacyBudget",
    # Memory
    "DifferentiableStack",
    "DifferentiableQueue",
    "DifferentiableDeque",
    "PriorityQueue",
    "DifferentiableStackEnsemble",
    "NeuralTuringMachine",
    "LookupFreeNTM",
    "create_ntm",
    "HopfieldAttention",
    "SparseMemoryAttention",
    "KeyValueMemoryAttention",
    "AssociativeAttention",
    "MemoryAugmentedAttention",
    "RoutingAttention",
    "SetAttention",
    "CosineSimilarityAddressing",
    "EuclideanDistanceAddressing",
    "DotProductAddressing",
    "LearnedSimilarityAddressing",
    "MultiHeadContentAddressing",
    "HybridAddressing",
    "AttentionBasedAddressing",
    "MemoryBank",
    "AssociativeMemory",
    "WorkingMemory",
    "EpisodicMemory",
    "SemanticMemory",
    "HierarchicalMemorySystem",
    "ContinualLearningMemory",
    "MetaLearningMemory",
]

try:
    from .privacy import (
        PrivacyEngine,
        FederatedPrivacyEngine,
        PrivacyEngineConfig,
        DPSGD,
        DPTrainer,
        GaussianMechanism,
        LaplaceMechanism,
        ExponentialMechanism,
        GaussianMixtureMechanism,
        RDPAccountant,
        BasicAccountant,
        GaussianAccountant,
        PrivacyBudgetTracker,
        StaticClipper,
        AdaptiveClipper,
        PerLayerClipper,
        NormAccountingClipper,
        SubsampleAmplifier,
        ShuffleAmplifier,
        LocalAmplifier,
        ComposeAmplifier,
        PoissonSampler,
        BatchSampler,
        StratifiedSampler,
        WeightedSampler,
        PrivacyAwareSampler,
        NoisyAggregator,
        ClippedAggregator,
        SecureAggregator,
        DPFederatedAggregator,
        SecureAggregationProtocol,
        AdditiveSecretSharing,
        ThresholdCryptography,
        SecureSum,
        create_privacy_engine,
        create_noise_mechanism,
        create_accountant,
        create_clipper,
        create_aggregator,
        create_sampler,
        create_secure_aggregation,
        amplify_privacy,
        compose_epsilons,
        compute_gaussian_epsilon,
        compute_laplace_epsilon,
        convert_rdp_to_dp,
        convert_dp_to_rdp,
        compute_subsampled_epsilon,
        compute_gaussian_composition,
        compute_privacy_budget,
        estimate_utilitarian_epsilon,
        compute_noise_scale,
        PrivacyAccountantSimple,
        PrivacyBudget,
        estimate_epsilon,
    )

    _PRIVACY_AVAILABLE = True
except ImportError:
    _PRIVACY_AVAILABLE = False

try:
    from .logic import (
        PropositionalLogic,
        DPLLSolver,
        SATProblem,
        Atom,
        Constant,
        Formula as PropFormula,
        UnifierFOL,
        Skolemizer,
        FOLToCNF,
        ModalSystem,
        KripkeFrame,
        KripkeModel,
        ModalFormula,
        ModalLogic,
        ModalTableauProver,
        Concept,
        Role,
        TBox,
        ABox,
        StructuralReasoner,
        TableauAlgorithm,
        CDCLSolver,
        ResolutionProver,
        FirstOrderClause,
        Literal,
        Clause,
        NaturalDeductionProver,
    )

    _LOGIC_AVAILABLE = True
except ImportError as e:
    print(f"Logic module import error: {e}")
    _LOGIC_AVAILABLE = False

try:
    from .generative import (
        DDPM,
        DiffusionScheduler,
        DDIM,
        DDIMScheduler,
        ScoreBasedModel,
        ScoreNetwork,
        AnnealedLangevinDynamics,
        LatentDiffusionModel,
        AutoencoderKL,
        VQModel,
        StyleGAN,
        StyleGAN2,
        MappingNetwork,
        SynthesisNetwork,
        StyleGAN2Generator,
        ProgressiveGrowing,
        BigGAN,
        ConditionalGenerator,
        ProjectionDiscriminator,
        AutoregressiveTransformer,
        PositionalEncoding,
        GPTGenerator,
        GPT2LMHeadModel,
        PixelCNN,
        PixelCNNPP,
        GatedMaskedConv2d,
        EnergyBasedModel,
        ConvEnergyModel,
        LangevinSampler,
        HMCSampler,
        EBMTrainer,
        FlowMatching,
        ConditionalFlowMatching,
        OptimalTransportFlow,
        SinkhornDivergence,
    )

    _GENERATIVE_AVAILABLE = True
except ImportError as e:
    print(f"Generative module import error: {e}")
    _GENERATIVE_AVAILABLE = False

try:
    from .finance import (
        FinancialTimeSeries,
        TechnicalIndicators,
        OHLCVData,
        FinancialFeatureEngineer,
        FinancialScaler,
        GARCH,
        EGARCH,
        GJR_GARCH,
        EWMAVolatility,
        RealizedVolatility,
        ImpliedVolatility,
        VolatilityForecast,
        StochasticVolatility,
        VolatilitySurface,
        ValueAtRisk,
        ConditionalVaR,
        MaximumDrawdown,
        PerformanceMetrics,
        RiskMetrics,
        FactorExposure,
        StressTest,
        RiskReport,
        MeanVarianceOptimization,
        RiskParity,
        BlackLitterman,
        MinimumVariance,
        MaximumSharpe,
        HierarchicalRiskParity,
        EqualWeightPortfolio,
        InverseVolatility,
        KellyCriterion,
        PortfolioBacktest,
        OrderType,
        OrderSide,
        OrderStatus,
        Order,
        Position,
        Portfolio,
        SignalGenerator,
        MovingAverageCrossover,
        RSIStrategy,
        MomentumStrategy,
        MeanReversionStrategy,
        Backtester,
        TradingStrategy,
        CNNLSTMTrading,
        AttentionTrading,
        ExecutionAlgorithm,
        RiskManager,
    )

    _FINANCE_AVAILABLE = True
except ImportError as e:
    print(f"Finance module import error: {e}")
    _FINANCE_AVAILABLE = False

try:
    from .scene_understanding import (
        compute_psnr,
        compute_ssim,
        normalize_tensor,
        safe_divide,
        gradient_x,
        gradient_y,
        IntermediateLayerGetter,
        meshgrid,
        get_gaussian_kernel,
        apply_bilateral_filter,
        SceneClassifier,
        ResNetSceneClassifier,
        MultiScaleSceneFeatures,
        VisionTransformerSceneClassifier,
        TransformerBlock,
        SceneContextEncoder,
        create_scene_classifier,
        SemanticSegmentationHead,
        PSPModule,
        SceneSegmentationNetwork,
        BoundaryAwareSegmentation,
        BoundaryRefinementModule,
        PanopticSegmentationHead,
        DeepLabV3Plus,
        ASPP,
        create_segmentation_model,
        DepthEncoder,
        DepthDecoder,
        MonocularDepthEstimator,
        ConfidenceDepthEstimator,
        DepthRefinementModule,
        MultiScaleDepthFusion,
        DispNet,
        ResidualDepthRefinement,
        ResidualBlock,
        create_depth_estimator,
        SurfaceNormalEncoder,
        NormalDecoder,
        SurfaceNormalEstimator,
        NormalRefinementModule,
        ConfidenceWeightedNormals,
        NormalFromDepthConsistency,
        EdgeAwareNormalSmoothing,
        MultiScaleNormalPrediction,
        NormalizationLoss,
        create_normal_estimator,
        ObjectDetector,
        RegionProposalNetwork,
        RCNNHead,
        RoIAlignPooling,
        RelationshipPredictor,
        SceneGraphBuilder,
        SceneGraphNode,
        SceneGraphEdge,
        SceneGraph,
        GraphConvolutionLayer,
        SceneGraphReasoning,
        RELATIONSHIP_CLASSES,
        create_scene_graph_model,
    )

    _SCENE_UNDERSTANDING_AVAILABLE = True
except ImportError as e:
    print(f"Scene understanding module import error: {e}")
    _SCENE_UNDERSTANDING_AVAILABLE = False
