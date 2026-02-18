"""
fishstick.bayesian_dl
====================

Bayesian Deep Learning module for fishstick framework.

Provides comprehensive tools for:
- Variational inference layers
- Bayesian linear regression
- Monte Carlo dropout
- Deep ensembles for uncertainty
- Bayesian model averaging
- Uncertainty metrics and ELBO computation
"""

from .variational_layers import (
    ConcreteDropout,
    VariationalLinear,
    VariationalConv2d,
    VariationalBatchNorm2d,
    FlipoutLinear,
    ReparameterizedDense,
    HeteroscedasticLoss,
)

from .bayesian_linear import (
    BayesianLinearRegression,
    SparseBayesianLinearRegression,
    RobustBayesianLinearRegression,
    MultiTaskBayesianLinearRegression,
    EmpiricalBayesLinearRegression,
)

from .mc_dropout import (
    MCDropout,
    DropoutAsBayes,
    ConcreteDropout as ConcreteDropoutMC,
    MCDropoutLinear,
    MCDropoutConv2d,
    DropoutSchedule,
    MCDropoutClassifier,
    DropoutUncertaintyMetrics,
)

from .deep_ensembles import (
    DeepEnsemble,
    WeightedDeepEnsemble,
    DiversityPromotingEnsemble,
    SnapshotEnsemble,
    FastGeometricEnsemble,
    EnsembleWithKnockout,
    BatchEnsemble,
    EnsembleCalibration,
    EnsembleUncertaintyMetrics,
)

from .model_averaging import (
    BayesianModelAveraging,
    ModelWeightOptimizer,
    HyperpriorBMA,
    MixtureOfExperts,
    ModelSelectionBMA,
    StackingBMA,
    OnlineBMA,
    BootstrapBMA,
    BMAContinuous,
)

from .bayesian_nn import (
    BayesianModule,
    BayesianConv2d,
    BayesianLinear,
    BayesianSequential,
    BayesianNeuralNetwork,
    LaplaceApproximation,
    SWAG,
    RadfordNeal,
)

from .variational_utils import (
    Prior,
    NormalPrior,
    LaplacePrior,
    HorseshoePrior,
    SpikeAndSlabPrior,
    VariationalPosterior,
    MeanFieldVariationalPosterior,
    KLDivergence,
    RenyiDivergence,
    VariationalLoss,
    PackagedKL,
    CholeskyVariationalPosterior,
    FlowVariationalPosterior,
    ScaleMixture,
    make_prior,
    make_posterior,
)

from .uncertainty_metrics import (
    UncertaintyMetrics,
    RegressionUncertaintyMetrics,
    UncertaintyDecomposition,
    OODDetectionMetrics,
    ConfidenceCalibration,
)

from .elbo import (
    ELBO,
    IWAE,
    WarmupELBO,
    SpectralELBO,
    DropoutELBO,
    CompositeELBO,
    VariationalEBLO,
    MonteCarloELBO,
    BayesianTripleLoss,
    make_elbo_loss,
    ELBOWithGradientClipping,
)

__all__ = [
    # Variational Layers
    "ConcreteDropout",
    "VariationalLinear",
    "VariationalConv2d",
    "VariationalBatchNorm2d",
    "FlipoutLinear",
    "ReparameterizedDense",
    "HeteroscedasticLoss",
    # Bayesian Linear Regression
    "BayesianLinearRegression",
    "SparseBayesianLinearRegression",
    "RobustBayesianLinearRegression",
    "MultiTaskBayesianLinearRegression",
    "EmpiricalBayesLinearRegression",
    # MC Dropout
    "MCDropout",
    "DropoutAsBayes",
    "ConcreteDropoutMC",
    "MCDropoutLinear",
    "MCDropoutConv2d",
    "DropoutSchedule",
    "MCDropoutClassifier",
    "DropoutUncertaintyMetrics",
    # Deep Ensembles
    "DeepEnsemble",
    "WeightedDeepEnsemble",
    "DiversityPromotingEnsemble",
    "SnapshotEnsemble",
    "FastGeometricEnsemble",
    "EnsembleWithKnockout",
    "BatchEnsemble",
    "EnsembleCalibration",
    "EnsembleUncertaintyMetrics",
    # Model Averaging
    "BayesianModelAveraging",
    "ModelWeightOptimizer",
    "HyperpriorBMA",
    "MixtureOfExperts",
    "ModelSelectionBMA",
    "StackingBMA",
    "OnlineBMA",
    "BootstrapBMA",
    "BMAContinuous",
    # Bayesian NN
    "BayesianModule",
    "BayesianConv2d",
    "BayesianLinear",
    "BayesianSequential",
    "BayesianNeuralNetwork",
    "LaplaceApproximation",
    "SWAG",
    "RadfordNeal",
    # Variational Utils
    "Prior",
    "NormalPrior",
    "LaplacePrior",
    "HorseshoePrior",
    "SpikeAndSlabPrior",
    "VariationalPosterior",
    "MeanFieldVariationalPosterior",
    "KLDivergence",
    "RenyiDivergence",
    "VariationalLoss",
    "PackagedKL",
    "CholeskyVariationalPosterior",
    "FlowVariationalPosterior",
    "ScaleMixture",
    "make_prior",
    "make_posterior",
    # Uncertainty Metrics
    "UncertaintyMetrics",
    "RegressionUncertaintyMetrics",
    "UncertaintyDecomposition",
    "OODDetectionMetrics",
    "ConfidenceCalibration",
    # ELBO
    "ELBO",
    "IWAE",
    "WarmupELBO",
    "SpectralELBO",
    "DropoutELBO",
    "CompositeELBO",
    "VariationalEBLO",
    "MonteCarloELBO",
    "BayesianTripleLoss",
    "make_elbo_loss",
    "ELBOWithGradientClipping",
]
