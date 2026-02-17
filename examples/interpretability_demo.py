#!/usr/bin/env python3
"""
Advanced Interpretability Examples

This script demonstrates the comprehensive interpretability tools
available in fishstick.
"""

import torch
import torch.nn as nn
import numpy as np

# Import fishstick interpretability tools
from fishstick.interpretability import (
    UnifiedExplainer,
    ExplainerPipeline,
    quick_explain,
    explain_and_visualize,
    # Individual methods
    SaliencyMap,
    IntegratedGradients,
    SmoothGrad,
    GradCAM,
    LayerwiseRelevancePropagation,
    AttentionVisualization,
)


def create_sample_model():
    """Create a sample CNN for demonstration."""

    class SampleCNN(nn.Module):
        def __init__(self):
            super().__init__()
            self.features = nn.Sequential(
                nn.Conv2d(3, 16, 3, padding=1),
                nn.ReLU(),
                nn.MaxPool2d(2),
                nn.Conv2d(16, 32, 3, padding=1),
                nn.ReLU(),
                nn.MaxPool2d(2),
            )
            self.classifier = nn.Sequential(
                nn.Flatten(),
                nn.Linear(32 * 8 * 8, 64),
                nn.ReLU(),
                nn.Linear(64, 10),
            )

        def forward(self, x):
            x = self.features(x)
            x = self.classifier(x)
            return x

    return SampleCNN()


def example_1_basic_attribution():
    """Example 1: Basic attribution methods."""
    print("=" * 70)
    print("Example 1: Basic Attribution Methods")
    print("=" * 70)

    # Create model and sample input
    model = create_sample_model()
    model.eval()

    # Random 32x32 RGB image
    image = torch.randn(1, 3, 32, 32)

    print(f"\nInput shape: {image.shape}")
    print(f"Model output shape: {model(image).shape}")

    # Method 1: Saliency Map
    print("\n1. Saliency Map (Vanilla Gradient):")
    saliency = SaliencyMap(model)
    attribution = saliency(image)
    print(f"   Attribution shape: {attribution.shape}")
    print(
        f"   Attribution stats: mean={attribution.mean():.4f}, max={attribution.max():.4f}"
    )

    # Method 2: Integrated Gradients
    print("\n2. Integrated Gradients:")
    ig = IntegratedGradients(model, steps=30)
    attribution = ig(image, target_class=5)
    print(f"   Attribution shape: {attribution.shape}")
    print(
        f"   Attribution stats: mean={attribution.mean():.4f}, max={attribution.max():.4f}"
    )

    # Method 3: SmoothGrad
    print("\n3. SmoothGrad:")
    smoothgrad = SmoothGrad(model, n_samples=20)
    attribution = smoothgrad(image, target_class=5)
    print(f"   Attribution shape: {attribution.shape}")
    print(
        f"   Attribution stats: mean={attribution.mean():.4f}, max={attribution.max():.4f}"
    )

    # Method 4: GradCAM
    print("\n4. GradCAM:")
    gradcam = GradCAM(model, target_layer="features.3")  # After second conv
    heatmap = gradcam(image, target_class=5)
    print(f"   Heatmap shape: {heatmap.shape}")
    print(f"   Heatmap stats: mean={heatmap.mean():.4f}, max={heatmap.max():.4f}")

    print("\n✓ Basic attribution methods completed!\n")


def example_2_unified_explainer():
    """Example 2: Using the Unified Explainer API."""
    print("=" * 70)
    print("Example 2: Unified Explainer API")
    print("=" * 70)

    model = create_sample_model()
    model.eval()

    image = torch.randn(1, 3, 32, 32)

    # Create unified explainer
    explainer = UnifiedExplainer(model)

    print("\nAvailable methods:", list(explainer.METHODS.keys()))

    # Single method explanation
    print("\n1. Single Method Explanation (Integrated Gradients):")
    result = explainer.explain(image, method="integrated_gradients", target=3, steps=50)
    print(f"   Method: {result['method']}")
    print(f"   Target: {result['target']}")
    print(f"   Attribution shape: {result['attribution'].shape}")

    # Get top features
    print("\n2. Top Feature Importance:")
    importance = explainer.get_feature_importance(
        image, method="saliency", target=3, top_k=5
    )
    print(f"   Top 5 feature indices: {importance['top_indices'][:5]}")
    print(f"   Top 5 importance values: {importance['top_values'][:5]}")

    # Attribution summary statistics
    print("\n3. Attribution Summary Statistics:")
    summary = explainer.get_attribution_summary(
        image, method="integrated_gradients", target=3
    )
    for key, value in summary.items():
        print(f"   {key}: {value:.6f}")

    print("\n✓ Unified explainer completed!\n")


def example_3_compare_methods():
    """Example 3: Comparing multiple explanation methods."""
    print("=" * 70)
    print("Example 3: Compare Multiple Methods")
    print("=" * 70)

    model = create_sample_model()
    model.eval()

    image = torch.randn(1, 3, 32, 32)

    explainer = UnifiedExplainer(model)

    # Compare multiple methods
    print("\nComparing attribution methods:")
    methods = ["saliency", "integrated_gradients", "smoothgrad", "deeplift"]

    results = explainer.compare_methods(image, methods=methods, target=5)

    for method, result in results.items():
        if "error" in result:
            print(f"   {method}: Error - {result['error']}")
        else:
            attr = result["attribution"]
            print(
                f"   {method:25s}: shape={str(attr.shape):15s}, mean={attr.mean():.4f}"
            )

    print("\n✓ Method comparison completed!\n")


def example_4_pipeline():
    """Example 4: Using the Explainer Pipeline."""
    print("=" * 70)
    print("Example 4: Explainer Pipeline")
    print("=" * 70)

    model = create_sample_model()
    model.eval()

    image = torch.randn(1, 3, 32, 32)

    # Create pipeline with multiple steps
    pipeline = ExplainerPipeline(model)

    # Add explanation steps
    pipeline.add_step("saliency", name="gradient_attribution")
    pipeline.add_step("integrated_gradients", name="ig_attribution", steps=30)
    pipeline.add_step("smoothgrad", name="smooth_attribution", n_samples=20)

    print("\nPipeline steps:", [step["name"] for step in pipeline.steps])

    # Run pipeline
    print("\nRunning pipeline...")
    results = pipeline.run(image, target=5)

    for name, result in results.items():
        print(f"   {name}: shape={result['attribution'].shape}")

    # Aggregate attributions
    print("\nAggregating attributions (mean)...")
    aggregated = pipeline.aggregate(image, target=5, aggregation="mean")
    print(f"   Aggregated attribution shape: {aggregated.shape}")
    print(
        f"   Aggregated stats: mean={aggregated.mean():.4f}, max={aggregated.max():.4f}"
    )

    print("\n✓ Pipeline completed!\n")


def example_5_quick_explain():
    """Example 5: Quick one-off explanations."""
    print("=" * 70)
    print("Example 5: Quick Explain Function")
    print("=" * 70)

    model = create_sample_model()
    model.eval()

    image = torch.randn(1, 3, 32, 32)

    print("\nQuick explanations (one-liners):")

    # Quick saliency
    attr = quick_explain(model, image, method="saliency")
    print(f"   Saliency:      shape={attr.shape}, max={attr.max():.4f}")

    # Quick integrated gradients
    attr = quick_explain(model, image, method="integrated_gradients", target=5)
    print(f"   Integrated Gradients: shape={attr.shape}, max={attr.max():.4f}")

    # Quick SmoothGrad
    attr = quick_explain(model, image, method="smoothgrad", target=5)
    print(f"   SmoothGrad:    shape={attr.shape}, max={attr.max():.4f}")

    print("\n✓ Quick explain completed!\n")


def example_6_batch_explanation():
    """Example 6: Batch explanations."""
    print("=" * 70)
    print("Example 6: Batch Explanations")
    print("=" * 70)

    model = create_sample_model()
    model.eval()

    # Batch of 4 images
    batch = torch.randn(4, 3, 32, 32)
    targets = torch.tensor([0, 3, 5, 7])

    explainer = UnifiedExplainer(model)

    print(f"\nExplaining batch of {batch.shape[0]} images...")

    results = explainer.explain_batch(
        batch, method="integrated_gradients", targets=targets
    )

    print(f"   Generated {len(results)} explanations")
    for i, result in enumerate(results):
        print(
            f"   Image {i}: target={result['target']}, attribution shape={result['attribution'].shape}"
        )

    print("\n✓ Batch explanation completed!\n")


def example_7_lrp():
    """Example 7: Layer-wise Relevance Propagation."""
    print("=" * 70)
    print("Example 7: Layer-wise Relevance Propagation (LRP)")
    print("=" * 70)

    model = create_sample_model()
    model.eval()

    image = torch.randn(1, 3, 32, 32)

    print("\nComputing LRP attribution...")
    lrp = LayerwiseRelevancePropagation(model)
    relevance = lrp(image, target_class=5)

    print(f"   Relevance shape: {relevance.shape}")
    print(f"   Relevance stats:")
    print(f"     - Positive relevance: {(relevance > 0).sum().item()} pixels")
    print(f"     - Negative relevance: {(relevance < 0).sum().item()} pixels")
    print(f"     - Mean relevance: {relevance.mean():.6f}")
    print(f"     - Relevance conservation check: {relevance.sum().item():.4f}")

    print("\n✓ LRP completed!\n")


def example_8_noise_tunnel():
    """Example 8: Noise Tunnel for smoothing any attribution method."""
    print("=" * 70)
    print("Example 8: Noise Tunnel")
    print("=" * 70)

    from fishstick.interpretability.attribution import NoiseTunnel, SaliencyMap

    model = create_sample_model()
    model.eval()

    image = torch.randn(1, 3, 32, 32)

    # Wrap saliency with noise tunnel
    print("\nApplying Noise Tunnel to Saliency...")
    saliency_fn = SaliencyMap(model)

    tunnel = NoiseTunnel(lambda x, t: saliency_fn(x, t), n_samples=10, noise_level=0.1)

    smoothed = tunnel(image, target_class=5)

    print(f"   Smoothed attribution shape: {smoothed.shape}")
    print(f"   Smoothed stats: mean={smoothed.mean():.4f}, std={smoothed.std():.4f}")

    # Compare with vanilla
    vanilla = saliency_fn(image, 5)
    print(f"\n   Vanilla saliency: std={vanilla.std():.4f}")
    print(f"   Smoothed saliency: std={smoothed.std():.4f}")
    print(f"   Noise reduction: {(1 - smoothed.std() / vanilla.std()) * 100:.1f}%")

    print("\n✓ Noise Tunnel completed!\n")


def run_all_examples():
    """Run all interpretability examples."""
    print("\n" + "=" * 70)
    print("FISHSTICK ADVANCED INTERPRETABILITY EXAMPLES")
    print("=" * 70 + "\n")

    examples = [
        ("Basic Attribution Methods", example_1_basic_attribution),
        ("Unified Explainer API", example_2_unified_explainer),
        ("Compare Multiple Methods", example_3_compare_methods),
        ("Explainer Pipeline", example_4_pipeline),
        ("Quick Explain", example_5_quick_explain),
        ("Batch Explanations", example_6_batch_explanation),
        ("Layer-wise Relevance Propagation", example_7_lrp),
        ("Noise Tunnel", example_8_noise_tunnel),
    ]

    for name, func in examples:
        try:
            func()
        except Exception as e:
            print(f"\n✗ Error in {name}: {str(e)}\n")
            import traceback

            traceback.print_exc()

    print("\n" + "=" * 70)
    print("ALL EXAMPLES COMPLETED!")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    run_all_examples()
