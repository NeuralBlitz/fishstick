# fishstick Interpretability Module

Advanced model interpretability and explainability tools for fishstick.

## Overview

The `interpretability` module provides state-of-the-art tools for explaining neural network predictions, including feature attribution, concept-based explanations, attention visualization, and a unified API for easy access to all methods.

## Features

### Attribution Methods
- **SaliencyMap**: Vanilla gradient-based attribution
- **IntegratedGradients**: Axiomatic path-integrated gradients
- **SmoothGrad**: Noise-reduced saliency maps
- **DeepLIFT**: Activation difference attribution
- **GradCAM**: Gradient-weighted class activation mapping
- **SHAPValues**: SHAP approximation using gradients
- **LIMEExplainer**: Local Interpretable Model-agnostic Explanations
- **OcclusionSensitivity**: Occlusion-based importance analysis
- **LayerwiseRelevancePropagation (LRP)**: Layer-wise backpropagation
- **NoiseTunnel**: Smooth any attribution method with noise

### Attention Visualization
- **AttentionVisualization**: Extract and visualize attention weights
- **AttentionRollout**: Multi-layer attention accumulation
- **AttentionPatternAnalysis**: Analyze attention patterns and entropy

### Concept-Based Explanations
- **TCAV**: Testing with Concept Activation Vectors
- **ConceptExtractor**: Extract concepts using PCA/ICA
- **ConceptBottleneck**: Interpretable concept bottleneck models
- **LinearProbe**: Train linear probes on representations

### Unified API
- **UnifiedExplainer**: Single interface for all methods
- **ExplainerPipeline**: Chain multiple explanation methods
- **quick_explain**: One-off explanation function
- **explain_and_visualize**: Generate explanations with visualizations

## Quick Start

### Using the Unified Explainer

```python
from fishstick.interpretability import UnifiedExplainer
import torch

# Create model and explainer
model = YourModel()
explainer = UnifiedExplainer(model)

# Single method explanation
result = explainer.explain(
    inputs=image,
    method='integrated_gradients',
    target=5
)
attribution = result['attribution']

# Compare multiple methods
comparisons = explainer.compare_methods(
    image,
    methods=['saliency', 'gradcam', 'integrated_gradients'],
    target=5
)
```

### Quick One-Off Explanations

```python
from fishstick.interpretability import quick_explain

# One line explanation
attribution = quick_explain(
    model, 
    image, 
    method='gradcam', 
    target=5
)
```

### Individual Methods

#### Integrated Gradients

```python
from fishstick.interpretability import IntegratedGradients

ig = IntegratedGradients(model, steps=50)
attribution = ig(image, target_class=5)
```

#### SmoothGrad

```python
from fishstick.interpretability import SmoothGrad

sg = SmoothGrad(model, n_samples=50, noise_level=0.15)
smooth_attribution = sg(image, target_class=5)
```

#### GradCAM

```python
from fishstick.interpretability import GradCAM

gradcam = GradCAM(model, target_layer='features.3')
heatmap = gradcam(image, target_class=5)
```

#### Layer-wise Relevance Propagation

```python
from fishstick.interpretability import LayerwiseRelevancePropagation

lrp = LayerwiseRelevancePropagation(model)
relevance = lrp(image, target_class=5)
```

### Pipeline for Multiple Methods

```python
from fishstick.interpretability import ExplainerPipeline

pipeline = ExplainerPipeline(model)
pipeline.add_step('saliency', name='grad_attribution')
pipeline.add_step('integrated_gradients', name='ig_attribution', steps=30)
pipeline.add_step('smoothgrad', name='smooth_attribution')

# Run all methods
results = pipeline.run(image, target=5)

# Aggregate attributions
aggregated = pipeline.aggregate(image, target=5, aggregation='mean')
```

### Batch Explanations

```python
# Explain multiple samples
batch = torch.randn(4, 3, 224, 224)  # 4 images
targets = torch.tensor([0, 3, 5, 7])

results = explainer.explain_batch(
    batch,
    method='integrated_gradients',
    targets=targets
)
```

### Feature Importance

```python
# Get top-k most important features
importance = explainer.get_feature_importance(
    image,
    method='integrated_gradients',
    target=5,
    top_k=10
)

print(f"Top indices: {importance['top_indices']}")
print(f"Top values: {importance['top_values']}")
```

### Attention Visualization

```python
from fishstick.interpretability import AttentionVisualization, AttentionRollout

# Extract attention weights
vis = AttentionVisualization(model)
attention_weights = vis.get_attention(text_tokens)

# Visualize specific head
head_attention = vis.visualize_head(layer=2, head=4)

# Compute attention rollout
rollout = AttentionRollout(model, num_layers=12)
attention_matrix = rollout.compute_rollout(text_tokens)

# Analyze patterns
from fishstick.interpretability import AttentionPatternAnalysis

entropy = AttentionPatternAnalysis.compute_entropy(attention_weights[0])
density = AttentionPatternAnalysis.compute_attention_density(attention_weights[0])
```

### Concept-Based Explanations (TCAV)

```python
from fishstick.interpretability import TCAV

tcav = TCAV(model)

# Define concept examples
concept_images = torch.stack([...])  # Images representing a concept
random_images = torch.stack([...])   # Random baseline images

# Compute TCAV score
score = tcav.compute_tcav_score(
    input_examples=test_images,
    concept_examples=concept_images,
    target_class=5,
    random_examples=random_images
)

print(f"TCAV Score: {score:.3f}")  # Fraction of inputs where concept matters
```

### Noise Tunnel for Smoothing

```python
from fishstick.interpretability import NoiseTunnel, SaliencyMap

# Wrap any attribution method with noise smoothing
saliency_fn = SaliencyMap(model)
tunnel = NoiseTunnel(
    lambda x, t: saliency_fn(x, t),
    n_samples=10,
    noise_level=0.1
)

smoothed = tunnel(image, target_class=5)
```

## Attribution Methods Reference

### Integrated Gradients
**Reference**: Sundararajan et al., "Axiomatic Attribution for Deep Networks", 2017

Computes the path integral of gradients along the straight line path from a baseline to the input.

```python
ig = IntegratedGradients(model, baseline=None, steps=50)
attribution = ig(image, target_class=5)
```

### SmoothGrad
**Reference**: Smilkov et al., "SmoothGrad: removing noise by adding noise", 2017

Reduces noise in saliency maps by averaging gradients over noisy samples.

```python
sg = SmoothGrad(model, n_samples=50, noise_level=0.15)
attribution = sg(image, target_class=5)
```

### DeepLIFT
**Reference**: Shrikumar et al., "Learning Important Features Through Propagating Activation Differences", 2017

Attributes importance scores by comparing activations to reference activations.

```python
dl = DeepLIFT(model, baseline=None)
attribution = dl(image, target_class=5)
```

### Layer-wise Relevance Propagation (LRP)
**Reference**: Bach et al., "On Pixel-Wise Explanations for Non-Linear Classifier Decisions", 2015

Propagates relevance backwards through the network layer by layer.

```python
lrp = LayerwiseRelevancePropagation(model, epsilon=1e-9)
relevance = lrp(image, target_class=5)
```

### GradCAM
**Reference**: Selvaraju et al., "Grad-CAM: Visual Explanations from Deep Networks", 2017

Gradient-weighted class activation mapping for visual explanations.

```python
gradcam = GradCAM(model, target_layer='layer4')
heatmap = gradcam(image, target_class=5)
```

## Dependencies

- `torch`: PyTorch for tensor operations
- `numpy`: Numerical operations
- `sklearn`: Linear models for LIME (optional)
- `matplotlib`: For visualization (optional)

## Examples

See `examples/interpretability_demo.py` for comprehensive usage examples:

```bash
python examples/interpretability_demo.py
```

This demonstrates:
1. Basic attribution methods
2. Unified explainer API
3. Comparing multiple methods
4. Explainer pipelines
5. Quick one-off explanations
6. Batch explanations
7. Layer-wise relevance propagation
8. Noise tunnel for smoothing

## API Reference

### UnifiedExplainer

```python
class UnifiedExplainer:
    def explain(inputs, method='saliency', target=None, **kwargs) -> Dict
    def compare_methods(inputs, methods=None, target=None, **kwargs) -> Dict
    def get_feature_importance(inputs, method='integrated_gradients', 
                               target=None, top_k=None, **kwargs) -> Dict
    def explain_batch(inputs, method='saliency', targets=None, **kwargs) -> List[Dict]
    def get_attribution_summary(inputs, method='integrated_gradients', 
                                target=None, **kwargs) -> Dict[str, float]
```

### ExplainerPipeline

```python
class ExplainerPipeline:
    def add_step(method, name=None, **kwargs) -> self
    def run(inputs, target=None) -> Dict[str, Dict]
    def aggregate(inputs, target=None, aggregation='mean') -> Tensor
```

## Best Practices

1. **Choose the right method**:
   - Use `SaliencyMap` for quick, simple explanations
   - Use `IntegratedGradients` for more reliable, axiomatic explanations
   - Use `SmoothGrad` to reduce noise in explanations
   - Use `GradCAM` for visual explanations of CNNs
   - Use `LRP` for layer-wise importance propagation

2. **Validate explanations**:
   - Compare multiple methods to ensure consistency
   - Use the `compare_methods` function to see agreement across methods
   - Check that important features align with domain knowledge

3. **Handle batch dimensions**:
   - Most methods return attributions with shape matching input
   - Some methods may squeeze batch dimensions - check output shapes
   - Use `explain_batch` for processing multiple samples efficiently

4. **Normalization**:
   - Attributions can be positive or negative
   - Use `abs()` for magnitude-only importance
   - Normalize for visualization or comparison across methods

## Citation

If you use this module in your research, please cite:

```bibtex
@software{fishstick_2026,
  title={fishstick: A Mathematically Rigorous AI Framework},
  url={https://github.com/NeuralBlitz/fishstick},
  year={2026}
}
```

## License

MIT License - see LICENSE file for details.
