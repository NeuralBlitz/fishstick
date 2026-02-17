"""
Unified Explainer API

High-level interface for all interpretability methods in fishstick.
"""

from typing import Optional, List, Dict, Union, Callable, Any
import torch
from torch import nn
import numpy as np

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


class UnifiedExplainer:
    """
    Unified interface for all interpretability methods.

    Provides a consistent API for generating explanations with multiple methods.

    Example:
        >>> explainer = UnifiedExplainer(model)
        >>>
        >>> # Single method explanation
        >>> explanation = explainer.explain(
        ...     inputs=image,
        ...     method='integrated_gradients',
        ...     target=5
        ... )
        >>>
        >>> # Compare multiple methods
        >>> comparisons = explainer.compare_methods(
        ...     inputs=image,
        ...     methods=['saliency', 'gradcam', 'integrated_gradients']
        ... )
    """

    # Registry of available methods
    METHODS: Dict[str, type] = {
        # Attribution methods
        "saliency": SaliencyMap,
        "integrated_gradients": IntegratedGradients,
        "shap": SHAPValues,
        "lime": LIMEExplainer,
        "gradcam": GradCAM,
        "occlusion": OcclusionSensitivity,
        "smoothgrad": SmoothGrad,
        "deeplift": DeepLIFT,
        "lrp": LayerwiseRelevancePropagation,
        # Attention methods
        "attention": AttentionVisualization,
        "attention_rollout": AttentionRollout,
        # Concept methods
        "tcav": TCAV,
    }

    def __init__(self, model: nn.Module):
        """
        Initialize the Unified Explainer.

        Args:
            model: PyTorch model to explain
        """
        self.model = model
        self.model.eval()
        self._explainers: Dict[str, Any] = {}
        self._default_params: Dict[str, Dict] = {
            "integrated_gradients": {"steps": 50},
            "smoothgrad": {"n_samples": 50, "noise_level": 0.15},
            "shap": {"background_size": 10},
            "occlusion": {"window_size": 8, "stride": 4},
        }

    def explain(
        self,
        inputs: torch.Tensor,
        method: str = "saliency",
        target: Optional[Union[int, torch.Tensor]] = None,
        **kwargs,
    ) -> Dict[str, Any]:
        """
        Generate explanation using specified method.

        Args:
            inputs: Input tensor (batch_size, ...)
            method: Explanation method name
            target: Target class index or tensor
            **kwargs: Method-specific arguments

        Returns:
            Dictionary containing:
                - 'attribution': Attribution map/tensor
                - 'method': Method name used
                - 'target': Target class
                - Additional method-specific outputs

        Raises:
            ValueError: If method is not recognized
        """
        if method not in self.METHODS:
            raise ValueError(
                f"Unknown method '{method}'. "
                f"Available methods: {list(self.METHODS.keys())}"
            )

        # Get or create explainer instance
        if method not in self._explainers:
            if method in ["attention", "attention_rollout", "tcav"]:
                self._explainers[method] = self.METHODS[method](self.model)
            else:
                self._explainers[method] = self.METHODS[method](self.model)

        explainer = self._explainers[method]

        # Merge default params with user-provided kwargs
        params = self._default_params.get(method, {}).copy()
        params.update(kwargs)

        # Handle target tensor vs int
        if isinstance(target, torch.Tensor):
            target = target.item() if target.numel() == 1 else target

        # Call appropriate method
        if method == "lime":
            attribution, feature_importance = explainer.explain(
                inputs, target_class=target, **params
            )
            return {
                "attribution": attribution,
                "feature_importance": feature_importance,
                "method": method,
                "target": target,
            }
        elif method == "gradcam":
            heatmap = explainer(inputs, target_class=target)
            return {
                "attribution": heatmap,
                "heatmap": heatmap,
                "method": method,
                "target": target,
            }
        elif method in ["attention", "attention_rollout"]:
            if method == "attention":
                attention_weights = explainer.get_attention(inputs)
                return {
                    "attribution": attention_weights[-1] if attention_weights else None,
                    "attention_weights": attention_weights,
                    "method": method,
                    "target": target,
                }
            else:  # attention_rollout
                rollout = explainer.compute_rollout(inputs)
                return {
                    "attribution": rollout,
                    "rollout": rollout,
                    "method": method,
                    "target": target,
                }
        else:
            attribution = explainer(inputs, target_class=target)
            return {
                "attribution": attribution,
                "method": method,
                "target": target,
            }

    def compare_methods(
        self,
        inputs: torch.Tensor,
        methods: Optional[List[str]] = None,
        target: Optional[Union[int, torch.Tensor]] = None,
        **kwargs,
    ) -> Dict[str, Dict[str, Any]]:
        """
        Compare multiple explanation methods on the same input.

        Args:
            inputs: Input tensor
            methods: List of method names to compare (default: common attribution methods)
            target: Target class
            **kwargs: Arguments passed to each method

        Returns:
            Dictionary mapping method names to their explanation results
        """
        if methods is None:
            methods = ["saliency", "integrated_gradients", "smoothgrad", "gradcam"]

        results = {}
        for method in methods:
            try:
                result = self.explain(inputs, method, target, **kwargs)
                results[method] = result
            except Exception as e:
                results[method] = {
                    "error": str(e),
                    "method": method,
                }

        return results

    def get_feature_importance(
        self,
        inputs: torch.Tensor,
        method: str = "integrated_gradients",
        target: Optional[Union[int, torch.Tensor]] = None,
        top_k: Optional[int] = None,
        absolute: bool = True,
        **kwargs,
    ) -> Dict[str, Any]:
        """
        Get top-k most important features.

        Args:
            inputs: Input tensor
            method: Attribution method to use
            target: Target class
            top_k: Number of top features to return (default: all)
            absolute: Use absolute values for ranking
            **kwargs: Additional arguments for the method

        Returns:
            Dictionary with feature indices, importance scores, and metadata
        """
        result = self.explain(inputs, method, target, **kwargs)
        attribution = result["attribution"]

        # Flatten attribution for ranking
        flat_attr = attribution.flatten()

        if absolute:
            flat_attr = torch.abs(flat_attr)

        # Get top-k
        if top_k is None:
            top_k = min(10, len(flat_attr))

        top_values, top_indices = torch.topk(flat_attr, top_k)

        return {
            "top_indices": top_indices,
            "top_values": top_values,
            "attribution": attribution,
            "method": method,
            "target": target,
        }

    def explain_batch(
        self,
        inputs: torch.Tensor,
        method: str = "saliency",
        targets: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> List[Dict[str, Any]]:
        """
        Explain a batch of inputs.

        Args:
            inputs: Batch of inputs (batch_size, ...)
            method: Explanation method
            targets: Target classes for each input (optional)
            **kwargs: Additional arguments

        Returns:
            List of explanation dictionaries, one per input
        """
        results = []

        for i in range(inputs.shape[0]):
            input_i = inputs[i : i + 1]
            target_i = targets[i].item() if targets is not None else None

            result = self.explain(input_i, method, target_i, **kwargs)
            results.append(result)

        return results

    def get_attribution_summary(
        self,
        inputs: torch.Tensor,
        method: str = "integrated_gradients",
        target: Optional[Union[int, torch.Tensor]] = None,
        **kwargs,
    ) -> Dict[str, float]:
        """
        Get summary statistics for attributions.

        Args:
            inputs: Input tensor
            method: Attribution method
            target: Target class
            **kwargs: Additional arguments

        Returns:
            Dictionary of summary statistics
        """
        result = self.explain(inputs, method, target, **kwargs)
        attr = result["attribution"]

        if torch.is_tensor(attr):
            return {
                "mean": attr.mean().item(),
                "std": attr.std().item(),
                "min": attr.min().item(),
                "max": attr.max().item(),
                "abs_mean": attr.abs().mean().item(),
                "sparsity": (attr == 0).float().mean().item(),
            }
        else:
            return {"error": "Attribution is not a tensor"}


class ExplainerPipeline:
    """
    Pipeline for combining multiple explanation methods.

    Useful for creating ensemble explanations or multi-view analysis.
    """

    def __init__(self, model: nn.Module):
        self.model = model
        self.explainer = UnifiedExplainer(model)
        self.steps: List[Dict[str, Any]] = []

    def add_step(self, method: str, name: Optional[str] = None, **kwargs):
        """Add an explanation step to the pipeline."""
        step = {
            "method": method,
            "name": name or method,
            "params": kwargs,
        }
        self.steps.append(step)
        return self

    def run(
        self, inputs: torch.Tensor, target: Optional[Union[int, torch.Tensor]] = None
    ) -> Dict[str, Any]:
        """
        Run the explanation pipeline.

        Args:
            inputs: Input tensor
            target: Target class

        Returns:
            Dictionary of all explanation results
        """
        results = {}

        for step in self.steps:
            result = self.explainer.explain(
                inputs, method=step["method"], target=target, **step["params"]
            )
            results[step["name"]] = result

        return results

    def aggregate(
        self,
        inputs: torch.Tensor,
        target: Optional[Union[int, torch.Tensor]] = None,
        aggregation: str = "mean",
    ) -> torch.Tensor:
        """
        Run pipeline and aggregate attributions.

        Args:
            inputs: Input tensor
            target: Target class
            aggregation: Aggregation method ('mean', 'sum', 'max', 'min')

        Returns:
            Aggregated attribution tensor
        """
        results = self.run(inputs, target)

        attributions = []
        for name, result in results.items():
            if "attribution" in result and torch.is_tensor(result["attribution"]):
                attr = result["attribution"]
                # Ensure consistent shape by squeezing batch dimension if present
                if attr.dim() > inputs.dim() - 1:
                    attr = attr.squeeze(0) if attr.shape[0] == 1 else attr
                # Normalize to [0, 1] for fair aggregation
                attr_min = attr.min()
                attr_max = attr.max()
                if attr_max > attr_min:
                    attr = (attr - attr_min) / (attr_max - attr_min + 1e-10)
                attributions.append(attr)

        if not attributions:
            raise ValueError("No valid attributions to aggregate")

        # Ensure all attributions have the same shape
        target_shape = attributions[0].shape
        normalized_attributions = []
        for attr in attributions:
            if attr.shape != target_shape:
                # Skip attributions with incompatible shapes
                continue
            normalized_attributions.append(attr)

        if not normalized_attributions:
            raise ValueError("No attributions with compatible shapes to aggregate")

        stacked = torch.stack(normalized_attributions)

        if aggregation == "mean":
            return stacked.mean(dim=0)
        elif aggregation == "sum":
            return stacked.sum(dim=0)
        elif aggregation == "max":
            return stacked.max(dim=0)[0]
        elif aggregation == "min":
            return stacked.min(dim=0)[0]
        else:
            raise ValueError(f"Unknown aggregation: {aggregation}")


def quick_explain(
    model: nn.Module,
    inputs: torch.Tensor,
    method: str = "integrated_gradients",
    target: Optional[Union[int, torch.Tensor]] = None,
    **kwargs,
) -> torch.Tensor:
    """
    Quick explanation function for one-off use.

    Args:
        model: PyTorch model
        inputs: Input tensor
        method: Explanation method
        target: Target class
        **kwargs: Additional arguments

    Returns:
        Attribution tensor

    Example:
        >>> attribution = quick_explain(model, image, method='gradcam', target=5)
    """
    explainer = UnifiedExplainer(model)
    result = explainer.explain(inputs, method, target, **kwargs)
    return result["attribution"]


def explain_and_visualize(
    model: nn.Module,
    inputs: torch.Tensor,
    method: str = "integrated_gradients",
    target: Optional[Union[int, torch.Tensor]] = None,
    save_path: Optional[str] = None,
    **kwargs,
):
    """
    Generate explanation and create visualization.

    Args:
        model: PyTorch model
        inputs: Input tensor
        method: Explanation method
        target: Target class
        save_path: Path to save visualization (optional)
        **kwargs: Additional arguments

    Returns:
        Tuple of (attribution, figure)
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        raise ImportError("matplotlib is required for visualization")

    explainer = UnifiedExplainer(model)
    result = explainer.explain(inputs, method, target, **kwargs)
    attribution = result["attribution"]

    # Create visualization
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Original input
    if inputs.dim() == 4:  # Image
        img = inputs[0].permute(1, 2, 0).cpu().numpy()
        if img.shape[2] == 1:
            img = img.squeeze(2)
        axes[0].imshow(img, cmap="gray" if img.ndim == 2 else None)
    else:
        axes[0].imshow(inputs[0].cpu().numpy(), aspect="auto")
    axes[0].set_title("Input")
    axes[0].axis("off")

    # Attribution
    if torch.is_tensor(attribution):
        attr = attribution.cpu().numpy()
        if attr.ndim == 3:
            attr = np.abs(attr).mean(axis=0)
        elif attr.ndim == 2 and attr.shape[0] == attr.shape[1]:
            # Attention matrix
            im = axes[1].imshow(attr, cmap="hot", aspect="auto")
            plt.colorbar(im, ax=axes[1])
        else:
            attr = np.abs(attr).flatten()
            axes[1].bar(range(len(attr)), attr)
            axes[1].set_title("Feature Importance")

    axes[1].set_title(f"Explanation ({method})")

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")

    return attribution, fig


__all__ = [
    "UnifiedExplainer",
    "ExplainerPipeline",
    "quick_explain",
    "explain_and_visualize",
]
