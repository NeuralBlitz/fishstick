"""
fishstick Interpretability Module

Advanced model interpretability and explainability tools.

This module provides state-of-the-art methods for explaining neural network
predictions including:

Attribution Methods:
    - SaliencyMap: Vanilla gradient-based attribution
    - IntegratedGradients: Path-integrated gradients for axiomatic attribution
    - SmoothGrad: Noise-reduced saliency maps
    - DeepLIFT: Activation difference attribution
    - SHAPValues: SHAP approximation using gradients
    - GradCAM: Gradient-weighted class activation mapping
    - OcclusionSensitivity: Occlusion-based importance
    - LayerwiseRelevancePropagation (LRP): Layer-wise backpropagation

Gradient Attribution (New):
    - VanillaGradients: Basic gradient-based saliency
    - SmoothGrad: Noise-robust attribution via averaging
    - IntegratedGradients: Path integral gradients
    - GuidedBackprop: ReLU-modified backpropagation
    - GradientxSign: Gradient times sign attribution
    - DeepLIFT: Reference-based attribution

Class Activation Maps (New):
    - GradCAM: Gradient-weighted CAM
    - GradCAMPlusPlus: Improved localization
    - ScoreCAM: Score-weighted CAM
    - EigenCAM: PCA-based visualization
    - LayerCAM: Layer-wise activation mapping
    - XGradCAM: Axiom-based GradCAM

Attention Visualization:
    - AttentionVisualization: Extract and visualize attention weights
    - AttentionRollout: Multi-layer attention accumulation
    - AttentionFlow: Flow-based attention aggregation
    - HeadImportance: Attention head importance analysis
    - AttentionPatternAnalyzer: Pattern and entropy analysis
    - AttentionGradient: Gradient-based attention viz

Feature Importance (New):
    - PermutationImportance: Feature permutation scoring
    - FeatureAblation: Ablation-based importance
    - SHAPExplainer: Shapley value approximation
    - KernelSHAP: Efficient SHAP estimation
    - LIMEExplainer: Local interpretable models
    - OcclusionSensitivity: Region occlusion analysis
    - FeatureInteraction: Interaction detection

Concept-Based Explanations:
    - TCAV: Testing with Concept Activation Vectors
    - ConceptDiscovery: Automatic concept discovery
    - ConceptBottleneck: Interpretable concept bottleneck models
    - ConceptWhitening: Concept-aligned whitening
    - ConceptAlignmentScore: Alignment evaluation
    - LinearProbe: Train linear probes on representations

Unified API:
    - UnifiedExplainer: Single interface for all methods
    - ExplainerPipeline: Chain multiple explanation methods
    - quick_explain: One-off explanation function
    - explain_and_visualize: Generate explanations with visualizations

Example Usage:
    >>> from fishstick.interpretability import IntegratedGradients, GradCAM, TCAV
    >>>
    >>> # Gradient-based attribution
    >>> ig = IntegratedGradients(model)
    >>> attributions = ig.attribute(input_tensor, target=0)
    >>>
    >>> # Class Activation Mapping
    >>> cam = GradCAM(model, target_layer=model.layer4[-1])
    >>> heatmap = cam.attribute(input_tensor, target=0)
    >>>
    >>> # Concept-based explanations
    >>> tcav = TCAV(model, layer=model.layer3)
    >>> tcav.learn_concept(concept_examples, random_examples, 'striped')
    >>> result = tcav.test_concept(input_tensor, 'striped', target_class=0)
"""

from fishstick.interpretability.attribution import (
    SaliencyMap,
    IntegratedGradients,
    SHAPValues,
    LIMEExplainer,
    GradCAM as AttributionGradCAM,
    OcclusionSensitivity,
    SmoothGrad,
    DeepLIFT,
    LayerwiseRelevancePropagation,
    NoiseTunnel,
)

from fishstick.interpretability.attention import (
    AttentionVisualization,
    AttentionRollout,
    AttentionPatternAnalysis,
)

from fishstick.interpretability.concepts import (
    ConceptExtractor,
    TCAV,
    ConceptBottleneck,
    LinearProbe,
)

from fishstick.interpretability.unified import (
    UnifiedExplainer,
    ExplainerPipeline,
    quick_explain,
    explain_and_visualize,
)

from fishstick.interpretability.gradient_attribution import (
    GradientAttributionBase,
    VanillaGradients,
    SmoothGrad as SmoothGradV2,
    IntegratedGradients as IntegratedGradientsV2,
    GuidedBackprop,
    GradCAM as GradientGradCAM,
    GradientxSign,
    DeepLIFT as DeepLIFTV2,
    create_gradient_attribution,
)

from fishstick.interpretability.cam import (
    CAMBase,
    GradCAM,
    GradCAMPlusPlus,
    ScoreCAM,
    EigenCAM,
    LayerCAM,
    XGradCAM,
    MultiLayerCAM,
    create_cam,
)

from fishstick.interpretability.attention_viz import (
    AttentionVisualizerBase,
    AttentionRollout as AttentionRolloutV2,
    AttentionFlow,
    HeadImportance,
    AttentionPatternAnalyzer,
    AttentionGradient,
    create_attention_viz,
)

from fishstick.interpretability.feature_importance import (
    FeatureImportanceBase,
    PermutationImportance,
    FeatureAblation,
    SHAPExplainer,
    KernelSHAP,
    LIMEExplainer as LIMEExplainerV2,
    TreeSHAP,
    OcclusionSensitivity as OcclusionSensitivityV2,
    FeatureInteraction,
    create_feature_importance,
)

from fishstick.interpretability.concept_activation import (
    ConceptActivationBase,
    TCAV as TCAVV2,
    ConceptDiscovery,
    ConceptBottleneckModel,
    ConceptWhitening,
    ConceptAlignmentScore,
    ConceptCompleteness,
    create_concept_method,
)


__all__ = [
    "SaliencyMap",
    "IntegratedGradients",
    "SHAPValues",
    "LIMEExplainer",
    "OcclusionSensitivity",
    "SmoothGrad",
    "DeepLIFT",
    "LayerwiseRelevancePropagation",
    "NoiseTunnel",
    "AttentionVisualization",
    "AttentionRollout",
    "AttentionPatternAnalysis",
    "ConceptExtractor",
    "TCAV",
    "ConceptBottleneck",
    "LinearProbe",
    "UnifiedExplainer",
    "ExplainerPipeline",
    "quick_explain",
    "explain_and_visualize",
    "GradientAttributionBase",
    "VanillaGradients",
    "GuidedBackprop",
    "GradientGradCAM",
    "GradientxSign",
    "create_gradient_attribution",
    "CAMBase",
    "GradCAM",
    "GradCAMPlusPlus",
    "ScoreCAM",
    "EigenCAM",
    "LayerCAM",
    "XGradCAM",
    "MultiLayerCAM",
    "create_cam",
    "AttentionVisualizerBase",
    "AttentionFlow",
    "HeadImportance",
    "AttentionPatternAnalyzer",
    "AttentionGradient",
    "create_attention_viz",
    "FeatureImportanceBase",
    "PermutationImportance",
    "FeatureAblation",
    "SHAPExplainer",
    "KernelSHAP",
    "TreeSHAP",
    "FeatureInteraction",
    "create_feature_importance",
    "ConceptActivationBase",
    "ConceptDiscovery",
    "ConceptBottleneckModel",
    "ConceptWhitening",
    "ConceptAlignmentScore",
    "ConceptCompleteness",
    "create_concept_method",
]
