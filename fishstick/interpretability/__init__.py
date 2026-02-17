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

Attention Visualization:
    - AttentionVisualization: Extract and visualize attention weights
    - AttentionRollout: Multi-layer attention accumulation
    - AttentionPatternAnalysis: Analyze attention patterns and entropy

Concept-Based Explanations:
    - TCAV: Testing with Concept Activation Vectors
    - ConceptExtractor: Extract concepts using PCA/ICA
    - ConceptBottleneck: Interpretable concept bottleneck models
    - LinearProbe: Train linear probes on representations

Unified API:
    - UnifiedExplainer: Single interface for all methods
    - ExplainerPipeline: Chain multiple explanation methods
    - quick_explain: One-off explanation function
    - explain_and_visualize: Generate explanations with visualizations

Example Usage:
    >>> from fishstick.interpretability import UnifiedExplainer, quick_explain
    >>>
    >>> # Unified explainer
    >>> explainer = UnifiedExplainer(model)
    >>> result = explainer.explain(image, method='integrated_gradients', target=5)
    >>>
    >>> # Quick explain
    >>> attribution = quick_explain(model, image, method='gradcam')
    >>>
    >>> # Compare methods
    >>> comparisons = explainer.compare_methods(
    ...     image,
    ...     methods=['saliency', 'gradcam', 'integrated_gradients']
    ... )
"""

from fishstick.interpretability.attribution import (
    SaliencyMap,
    IntegratedGradients,
    SHAPValues,
    LIMEExplainer,
    GradCAM,
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

__all__ = [
    # Attribution
    "SaliencyMap",
    "IntegratedGradients",
    "SHAPValues",
    "LIMEExplainer",
    "GradCAM",
    "OcclusionSensitivity",
    "SmoothGrad",
    "DeepLIFT",
    "LayerwiseRelevancePropagation",
    "NoiseTunnel",
    # Attention
    "AttentionVisualization",
    "AttentionRollout",
    "AttentionPatternAnalysis",
    # Concepts
    "ConceptExtractor",
    "TCAV",
    "ConceptBottleneck",
    "LinearProbe",
    # Unified API
    "UnifiedExplainer",
    "ExplainerPipeline",
    "quick_explain",
    "explain_and_visualize",
]
