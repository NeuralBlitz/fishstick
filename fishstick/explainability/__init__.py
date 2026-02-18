"""
fishstick Explainability Module (XAI)

Advanced Explainable AI tools for understanding neural network predictions.

This module provides comprehensive explainability methods:

SHAP Values:
    - KernelSHAP: Model-agnostic Shapley value approximation
    - GradientSHAP: Gradient-based efficient SHAP
    - DeepSHAP: Layer-wise SHAP decomposition
    - TreeSHAP: Exact SHAP for tree-based models

Integrated Gradients:
    - IntegratedGradients: Path-integral gradient attribution
    - LayerIntegratedGradients: Layer-wise IG
    - SmoothedIntegratedGradients: Noise-reduced IG

Attention Visualization:
    - TransformerAttentionExtractor: Extract transformer attention
    - AttentionHeadAnalyzer: Analyze attention head importance
    - AttentionPatternClusterer: Cluster attention patterns
    - CrossAttentionVisualizer: Visualize encoder-decoder attention
    - AttentionGradientExtractor: Gradient through attention

Concept-Based Explanations:
    - LinearCAV: Learn concept activation vectors
    - TCAV: Testing with CAVs
    - ACEConceptDiscovery: Automatic concept discovery
    - ConceptBottleneckModel: Interpretable concept layers
    - ConceptWhitening: Align representations with concepts

Counterfactual Explanations:
    - GrowingSpheres: Sphere expansion counterfactuals
    - DiCEGenerator: Diverse counterfactual explanations
    - ProtoPFGenerator: Prototype-based counterfactuals
    - ActionableCounterfactual: Constrained counterfactuals

Utilities:
    - Attribution normalization and processing
    - Visualization data preparation
    - Explanation formatting and export

Metrics:
    - FidelityMetrics: AUC, insertion/deletion
    - ComplexityMetrics: Sparsity, compactness
    - StabilityMetrics: Sensitivity, robustness
    - CorrelationMetrics: Pearson, Spearman

Example Usage:
    >>> from fishstick.explainability import (
    ...     IntegratedGradients,
    ...     KernelSHAP,
    ...     TCAV,
    ...     GrowingSpheres,
    ... )
    >>>
    >>> # Feature attribution
    >>> ig = IntegratedGradients(model)
    >>> attributions = ig.attribute(input_tensor, target=0)
    >>>
    >>> # Concept explanations
    >>> cav = LinearCAV(model, layer)
    >>> concept = cav.learn_concept('striped', pos_samples, neg_samples)
    >>>
    >>> # Counterfactuals
    >>> cf_gen = GrowingSpheres(model)
    >>> cf = cf_gen.generate(input_tensor, target_class=1)
"""

from fishstick.explainability.shap_values import (
    ShapleyEstimator,
    KernelSHAP,
    GradientSHAP,
    DeepSHAP,
    TreeSHAP,
    ShapleySampler,
    create_shap_explainer,
)

from fishstick.explainability.integrated_gradients import (
    PathMethod,
    BaselineStrategy,
    IntegratedGradients,
    LayerIntegratedGradients,
    IntegratedGradientsWrapper,
    SmoothedIntegratedGradients,
    create_integrated_gradients,
)

from fishstick.explainability.attention_viz_xai import (
    AttentionType,
    AttentionWeights,
    TransformerAttentionExtractor,
    AttentionHeadAnalyzer,
    AttentionPatternClusterer,
    CrossAttentionVisualizer,
    AttentionGradientExtractor,
    create_attention_explainer,
)

from fishstick.explainability.concept_explanations import (
    Concept,
    ConceptActivationVector,
    LinearCAV,
    TCAV,
    ACEConceptDiscovery,
    ConceptBottleneckModel,
    ConceptWhitening,
    ConceptAlignmentScore,
    create_concept_explainer,
)

from fishstick.explainability.counterfactuals import (
    Counterfactual,
    CounterfactualGenerator,
    GrowingSpheres,
    DiCEGenerator,
    ProtoPFGenerator,
    ActionableCounterfactual,
    create_counterfactual_generator,
)

from fishstick.explainability.explanation_utils import (
    normalize_attributions,
    compute_attribution_mask,
    aggregate_attributions,
    convert_to_heatmap_format,
    format_explanation,
    export_explanation_json,
    compute_feature_importance_ranking,
    get_attribution_stats,
    smooth_attributions,
    compute_perturbation_curve,
    compute_sparsity_metrics,
    create_attribution_visualization_data,
)

from fishstick.explainability.xai_metrics import (
    FidelityMetrics,
    ComplexityMetrics,
    StabilityMetrics,
    CorrelationMetrics,
    ExplanationMetrics,
    create_metric_evaluator,
)


__all__ = [
    "ShapleyEstimator",
    "KernelSHAP",
    "GradientSHAP",
    "DeepSHAP",
    "TreeSHAP",
    "ShapleySampler",
    "create_shap_explainer",
    "PathMethod",
    "BaselineStrategy",
    "IntegratedGradients",
    "LayerIntegratedGradients",
    "IntegratedGradientsWrapper",
    "SmoothedIntegratedGradients",
    "create_integrated_gradients",
    "AttentionType",
    "AttentionWeights",
    "TransformerAttentionExtractor",
    "AttentionHeadAnalyzer",
    "AttentionPatternClusterer",
    "CrossAttentionVisualizer",
    "AttentionGradientExtractor",
    "create_attention_explainer",
    "Concept",
    "ConceptActivationVector",
    "LinearCAV",
    "TCAV",
    "ACEConceptDiscovery",
    "ConceptBottleneckModel",
    "ConceptWhitening",
    "ConceptAlignmentScore",
    "create_concept_explainer",
    "Counterfactual",
    "CounterfactualGenerator",
    "GrowingSpheres",
    "DiCEGenerator",
    "ProtoPFGenerator",
    "ActionableCounterfactual",
    "create_counterfactual_generator",
    "normalize_attributions",
    "compute_attribution_mask",
    "aggregate_attributions",
    "convert_to_heatmap_format",
    "format_explanation",
    "export_explanation_json",
    "compute_feature_importance_ranking",
    "get_attribution_stats",
    "smooth_attributions",
    "compute_perturbation_curve",
    "compute_sparsity_metrics",
    "create_attribution_visualization_data",
    "FidelityMetrics",
    "ComplexityMetrics",
    "StabilityMetrics",
    "CorrelationMetrics",
    "ExplanationMetrics",
    "create_metric_evaluator",
]
