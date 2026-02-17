"""
Causal Inference and Structural Causal Models.

Advanced causal inference tools including:
- Structural causal model implementations
- Do-calculus operators
- Causal discovery algorithms
- Counterfactual reasoning
- Causal effect estimation
- Sensitivity analysis
"""

from typing import Dict, List, Optional, Tuple, Union, Callable, Any
import torch
from torch import Tensor
import numpy as np
from dataclasses import dataclass, field

from .structural_models import (
    AdditiveSCM,
    NonlinearSCM,
    GaussianProcessSCM,
    LinearSCM,
    SCMEnsemble,
    NoiseDistribution,
    SCMTrainer,
)

from .do_calculus import (
    DoCalculus,
    BackDoorCriterion,
    FrontDoorCriterion,
    ConditionalIntervention,
    DoCalculusRules,
    identify_causal_effect,
    is_adjustment_set_valid,
)

from .interventions import (
    Intervention,
    InterventionType,
    AtomicIntervention,
    StochasticIntervention,
    ShiftIntervention,
    PolicyIntervention,
    InterventionGraph,
    intervened_graph,
    do_operator,
)

from .discovery import (
    CausalDiscoveryAlgorithm,
    PCAlgorithm,
    NOTEARS,
    FCIAlgorithm,
    GESAlgorithm,
    LiNGAM,
    CAM,
    CI_test,
    conditional_independence_test,
    skeleton_recovery,
    orient_edges,
)

from .dependency_graph import (
    CausalDAG,
    CausalGraphAnalyzer,
    AncestorQuery,
    d_separation,
    is_valid_adjustment_set,
    minimal_adjustment_set,
    markov_blanket,
    causal_ordering,
)

from .counterfactuals import (
    CounterfactualEngine,
    TwinNetworkMethod,
    StructuralMethod,
    CounterfactualOutcome,
    compute_counterfactual,
    counterfactual_uncertainty,
    natural_direct_effect,
    natural_indirect_effect,
    mediated_effect,
)

from .treatment_effects import (
    TreatmentEffectEstimator,
    CATEEstimator,
    HTEValidator,
    MetaLearner,
    SLearner,
    TLearner,
    XLearner,
    DRLearner,
    CausalForest,
    uplift_model,
    average_treatment_effect,
    conditional_average_treatment_effect,
    att,
)

from .effect_estimators import (
    EstimatorType,
    LinearRegressionEstimator,
    DifferenceInDifferences,
    RegressionDiscontinuity,
    InstrumentalVariableEstimator,
    FrontDoorEstimator,
    MarginalStructuralModel,
    GFormula,
    IPWEstimator,
    AIPWEstimator,
    causal_calibration,
)

from .sensitivity import (
    SensitivityAnalyzer,
    RosenbaumBounds,
    EValue,
    SensitivityParams,
    unmeasured_confounding_bounds,
    robustness_value,
    tip_point,
    confounder_strength,
)


__all__ = [
    # Structural models
    "AdditiveSCM",
    "NonlinearSCM",
    "GaussianProcessSCM",
    "LinearSCM",
    "SCMEnsemble",
    "NoiseDistribution",
    "SCMTrainer",
    # Do-calculus
    "DoCalculus",
    "BackDoorCriterion",
    "FrontDoorCriterion",
    "ConditionalIntervention",
    "DoCalculusRules",
    "identify_causal_effect",
    "is_adjustment_set_valid",
    # Interventions
    "Intervention",
    "InterventionType",
    "AtomicIntervention",
    "StochasticIntervention",
    "ShiftIntervention",
    "PolicyIntervention",
    "InterventionGraph",
    "intervened_graph",
    "do_operator",
    # Discovery
    "CausalDiscoveryAlgorithm",
    "PCAlgorithm",
    "NOTEARS",
    "FCIAlgorithm",
    "GESAlgorithm",
    "LiNGAM",
    "CAM",
    "CI_test",
    "conditional_independence_test",
    "skeleton_recovery",
    "orient_edges",
    # Dependency graph
    "CausalDAG",
    "CausalGraphAnalyzer",
    "AncestorQuery",
    "d_separation",
    "is_valid_adjustment_set",
    "minimal_adjustment_set",
    "markov_blanket",
    "causal_ordering",
    # Counterfactuals
    "CounterfactualEngine",
    "TwinNetworkMethod",
    "StructuralMethod",
    "CounterfactualOutcome",
    "compute_counterfactual",
    "counterfactual_uncertainty",
    "natural_direct_effect",
    "natural_indirect_effect",
    "mediated_effect",
    # Treatment effects
    "TreatmentEffectEstimator",
    "CATEEstimator",
    "HTEValidator",
    "MetaLearner",
    "SLearner",
    "TLearner",
    "XLearner",
    "DRLearner",
    "CausalForest",
    "uplift_model",
    "average_treatment_effect",
    "conditional_average_treatment_effect",
    "att",
    # Effect estimators
    "EstimatorType",
    "LinearRegressionEstimator",
    "DifferenceInDifferences",
    "RegressionDiscontinuity",
    "InstrumentalVariableEstimator",
    "FrontDoorEstimator",
    "MarginalStructuralModel",
    "GFormula",
    "IPWEstimator",
    "AIPWEstimator",
    "causal_calibration",
    # Sensitivity
    "SensitivityAnalyzer",
    "RosenbaumBounds",
    "EValue",
    "SensitivityParams",
    "unmeasured_confounding_bounds",
    "robustness_value",
    "tip_point",
    "confounder_strength",
]
