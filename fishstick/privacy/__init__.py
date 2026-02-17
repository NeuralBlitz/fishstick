"""
fishstick Privacy Module

Differential privacy and secure learning tools for building
privacy-preserving machine learning systems.

Modules:
    - noise: Noise addition mechanisms (Gaussian, Laplace, Exponential)
    - accountant: Privacy budget accounting (RDP, Basic)
    - clipping: Gradient clipping strategies
    - dp_sgd: DP-SGD optimizer and training
    - amplification: Privacy amplification techniques
    - sampling: Sampling methods for DP
    - aggregation: Private aggregation methods
    - secure_aggregation: Secure aggregation protocols
    - privacy_engine: High-level privacy training API
    - accounting_utils: Privacy accounting utilities

Example:
    >>> from fishstick.privacy import PrivacyEngine, DPSGD, GaussianMechanism
    >>>
    >>> engine = PrivacyEngine(model, epsilon=8.0, delta=1e-5)
    >>> history = engine.train(train_loader, epochs=10)
"""

from fishstick.privacy.noise import (
    NoiseMechanism,
    GaussianMechanism,
    LaplaceMechanism,
    ExponentialMechanism,
    GaussianMixtureMechanism,
    NoiseConfig,
    create_noise_mechanism,
)

from fishstick.privacy.accountant import (
    PrivacyAccountant,
    RDPAccountant,
    BasicAccountant,
    GaussianAccountant,
    PrivacyBudgetTracker,
    PrivacyAccount,
    create_accountant,
)

from fishstick.privacy.clipping import (
    GradientClipper,
    StaticClipper,
    AdaptiveClipper,
    PerLayerClipper,
    NormAccountingClipper,
    create_clipper,
)

from fishstick.privacy.dp_sgd import (
    DPSGD,
    DPTrainer,
    DPGradientDescent,
    DPConfig,
    create_dp_optimizer,
    compute_noise_batch,
    estimate_epsilon,
)

from fishstick.privacy.amplification import (
    PrivacyAmplifier,
    SubsampleAmplifier,
    ShuffleAmplifier,
    LocalAmplifier,
    ComposeAmplifier,
    AmplificationResult,
    amplify_privacy,
    compute_subsampled_epsilon,
    get_amplification_factor,
)

from fishstick.privacy.sampling import (
    SamplingConfig,
    PoissonSampler,
    BatchSampler,
    StratifiedSampler,
    WeightedSampler,
    PrivacyAwareSampler,
    create_sampler,
    compute_effective_sample_rate,
)

from fishstick.privacy.aggregation import (
    PrivateAggregator,
    NoisyAggregator,
    ClippedAggregator,
    SecureAggregator,
    DPFederatedAggregator,
    create_aggregator,
    compute_adaptive_noise,
)

from fishstick.privacy.secure_aggregation import (
    SecureAggregationProtocol,
    AdditiveSecretSharing,
    ThresholdCryptography,
    SecureSum,
    ClientState,
    AggregationResult,
    create_secure_aggregation,
)

from fishstick.privacy.privacy_engine import (
    PrivacyEngine,
    FederatedPrivacyEngine,
    PrivacyEngineConfig,
    create_privacy_engine,
    estimate_training_epsilon,
)

from fishstick.privacy.accounting_utils import (
    PrivacyBudget,
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
    compute_clipping_factor,
    compute_empirical_epsilon,
)

__all__ = [
    # Noise mechanisms
    "NoiseMechanism",
    "GaussianMechanism",
    "LaplaceMechanism",
    "ExponentialMechanism",
    "GaussianMixtureMechanism",
    "NoiseConfig",
    "create_noise_mechanism",
    # Accountancy
    "PrivacyAccountant",
    "RDPAccountant",
    "BasicAccountant",
    "GaussianAccountant",
    "PrivacyBudgetTracker",
    "PrivacyAccount",
    "create_accountant",
    # Clipping
    "GradientClipper",
    "StaticClipper",
    "AdaptiveClipper",
    "PerLayerClipper",
    "NormAccountingClipper",
    "create_clipper",
    # DP-SGD
    "DPSGD",
    "DPTrainer",
    "DPGradientDescent",
    "DPConfig",
    "create_dp_optimizer",
    "compute_noise_batch",
    "estimate_epsilon",
    # Amplification
    "PrivacyAmplifier",
    "SubsampleAmplifier",
    "ShuffleAmplifier",
    "LocalAmplifier",
    "ComposeAmplifier",
    "AmplificationResult",
    "amplify_privacy",
    "compute_subsampled_epsilon",
    "get_amplification_factor",
    # Sampling
    "SamplingConfig",
    "PoissonSampler",
    "BatchSampler",
    "StratifiedSampler",
    "WeightedSampler",
    "PrivacyAwareSampler",
    "create_sampler",
    "compute_effective_sample_rate",
    # Aggregation
    "PrivateAggregator",
    "NoisyAggregator",
    "ClippedAggregator",
    "SecureAggregator",
    "DPFederatedAggregator",
    "create_aggregator",
    "compute_adaptive_noise",
    # Secure Aggregation
    "SecureAggregationProtocol",
    "AdditiveSecretSharing",
    "ThresholdCryptography",
    "SecureSum",
    "ClientState",
    "AggregationResult",
    "create_secure_aggregation",
    # Privacy Engine
    "PrivacyEngine",
    "FederatedPrivacyEngine",
    "PrivacyEngineConfig",
    "create_privacy_engine",
    "estimate_training_epsilon",
    # Accounting Utils
    "PrivacyBudget",
    "compose_epsilons",
    "compute_gaussian_epsilon",
    "compute_laplace_epsilon",
    "convert_rdp_to_dp",
    "convert_dp_to_rdp",
    "compute_gaussian_composition",
    "compute_privacy_budget",
    "estimate_utilitarian_epsilon",
    "compute_noise_scale",
    "PrivacyAccountantSimple",
    "compute_clipping_factor",
    "compute_empirical_epsilon",
]
