"""
Explanation Utilities

Utility functions for explainable AI, including:
- Attribution normalization and processing
- Visualization helpers
- Explanation formatting and export
- Fidelity metrics computation
"""

from __future__ import annotations

from typing import Optional, List, Dict, Tuple, Union, Callable
import json

import torch
from torch import Tensor, nn
import numpy as np


def normalize_attributions(
    attributions: Tensor,
    method: str = "l2",
    epsilon: float = 1e-8,
) -> Tensor:
    """Normalize attribution values using various methods.

    Args:
        attributions: Attribution tensor
        method: Normalization method ('l2', 'l1', 'minmax', 'zscore')
        epsilon: Small constant for numerical stability

    Returns:
        Normalized attributions
    """
    if method == "l2":
        norm = torch.norm(attributions) + epsilon
        return attributions / norm

    elif method == "l1":
        norm = torch.norm(attributions, p=1) + epsilon
        return attributions / norm

    elif method == "minmax":
        min_val = attributions.min()
        max_val = attributions.max()
        if max_val - min_val < epsilon:
            return torch.zeros_like(attributions)
        return (attributions - min_val) / (max_val - min_val)

    elif method == "zscore":
        mean = attributions.mean()
        std = attributions.std() + epsilon
        return (attributions - mean) / std

    else:
        return attributions


def compute_attribution_mask(
    attributions: Tensor,
    threshold: Union[float, str] = "auto",
    top_k: Optional[int] = None,
) -> Tensor:
    """Compute binary mask from attributions.

    Args:
        attributions: Attribution tensor
        threshold: Threshold value or 'auto' for automatic
        top_k: If set, keep top k attributions

    Returns:
        Binary mask tensor
    """
    abs_attr = attributions.abs()

    if top_k is not None:
        mask = torch.zeros_like(abs_attr)
        flat = abs_attr.view(-1)
        if top_k > 0 and top_k < len(flat):
            threshold_value = flat.kthvalue(len(flat) - top_k)[0]
            mask = (abs_attr >= threshold_value).float()
        return mask

    if isinstance(threshold, str) and threshold == "auto":
        threshold = compute_auto_threshold(abs_attr)

    mask = (abs_attr > threshold).float()
    return mask


def compute_auto_threshold(attributions: Tensor) -> float:
    """Compute automatic threshold using Otsu's method approximation."""
    flat = attributions.flatten().cpu().numpy()
    hist, bin_edges = np.histogram(flat, bins=50)

    hist = hist.astype(float)
    hist = hist + 1e-8

    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

    total = hist.sum()
    weight1 = np.cumsum(hist)
    weight2 = total - weight1

    mean1 = np.cumsum(hist * bin_centers) / (weight1 + 1e-8)
    mean2 = (np.cumsum(hist * bin_centers)[::-1] / (weight2 + 1e-8))[::-1]

    variance = weight1 * weight2 * (mean1 - mean2) ** 2

    best_idx = np.argmax(variance)
    threshold = bin_centers[best_idx]

    return float(threshold)


def aggregate_attributions(
    attributions: Tensor,
    method: str = "sum",
    dim: Optional[int] = None,
) -> Tensor:
    """Aggregate attributions across dimensions.

    Args:
        attributions: Attribution tensor
        method: Aggregation method ('sum', 'mean', 'max', 'l2')
        dim: Dimension to aggregate over

    Returns:
        Aggregated attributions
    """
    if dim is None:
        dim = tuple(range(1, attributions.dim()))

    if method == "sum":
        return attributions.sum(dim=dim)
    elif method == "mean":
        return attributions.mean(dim=dim)
    elif method == "max":
        return attributions.max(dim=dim)[0]
    elif method == "l2":
        return torch.norm(attributions, p=2, dim=dim)
    else:
        return attributions


def convert_to_heatmap_format(
    attributions: Tensor,
    shape: Optional[Tuple[int, ...]] = None,
) -> np.ndarray:
    """Convert attributions to heatmap visualization format.

    Args:
        attributions: Attribution tensor
        shape: Target shape for reshaping

    Returns:
        Numpy array suitable for heatmap visualization
    """
    result = attributions.cpu()

    if shape is not None:
        result = result.reshape(shape)

    if result.dim() == 4:
        result = result.mean(dim=1)

    if result.dim() == 3:
        result = result.mean(dim=0)

    result = result.abs()
    result = result / (result.max() + 1e-8)

    return result.numpy()


def format_explanation(
    attributions: Tensor,
    feature_names: Optional[List[str]] = None,
    top_k: int = 10,
    include_values: bool = True,
) -> Dict:
    """Format attributions as human-readable explanation.

    Args:
        attributions: Attribution tensor
        feature_names: Names for each feature
        top_k: Number of top features to include
        include_values: Whether to include actual values

    Returns:
        Formatted explanation dictionary
    """
    flat_attr = attributions.flatten()

    if feature_names is None:
        feature_names = [f"feature_{i}" for i in range(len(flat_attr))]

    attr_with_names = list(zip(feature_names, flat_attr.cpu().numpy()))

    sorted_attrs = sorted(attr_with_names, key=lambda x: abs(x[1]), reverse=True)

    result = {
        "top_features": [],
        "total_positive": float(flat_attr[flat_attr > 0].sum().item()),
        "total_negative": float(flat_attr[flat_attr < 0].sum().item()),
        "total_attribution": float(flat_attr.sum().item()),
    }

    for name, value in sorted_attrs[:top_k]:
        entry = {"feature": name}
        if include_values:
            entry["importance"] = float(value)
        result["top_features"].append(entry)

    return result


def export_explanation_json(
    explanation: Dict,
    filepath: str,
    include_metadata: bool = True,
):
    """Export explanation to JSON file.

    Args:
        explanation: Formatted explanation dictionary
        filepath: Output file path
        include_metadata: Whether to include metadata
    """
    metadata = {
        "explanation_version": "1.0",
        "format": "json",
    }

    if include_metadata:
        explanation["_metadata"] = metadata

    with open(filepath, "w") as f:
        json.dump(explanation, f, indent=2)


def compute_feature_importance_ranking(
    attributions: Tensor,
    descending: bool = True,
) -> List[Tuple[int, float]]:
    """Compute feature importance ranking.

    Args:
        attributions: Attribution tensor
        descending: Whether to sort descending

    Returns:
        List of (feature_index, importance) tuples
    """
    flat = attributions.flatten()
    magnitudes = flat.abs()

    indices = magnitudes.argsort(descending=descending).tolist()

    return [(idx, magnitudes[idx].item()) for idx in indices]


def get_attribution_stats(
    attributions: Tensor,
) -> Dict[str, float]:
    """Compute statistical metrics for attributions.

    Args:
        attributions: Attribution tensor

    Returns:
        Dictionary of statistics
    """
    flat = attributions.flatten()

    return {
        "mean": float(flat.mean().item()),
        "std": float(flat.std().item()),
        "min": float(flat.min().item()),
        "max": float(flat.max().item()),
        "abs_mean": float(flat.abs().mean().item()),
        "abs_sum": float(flat.abs().sum().item()),
        "positive_count": int((flat > 0).sum().item()),
        "negative_count": int((flat < 0).sum().item()),
        "zero_count": int((flat == 0).sum().item()),
    }


def smooth_attributions(
    attributions: Tensor,
    kernel_size: int = 3,
    sigma: float = 1.0,
) -> Tensor:
    """Apply Gaussian smoothing to attributions.

    Args:
        attributions: Attribution tensor
        kernel_size: Size of Gaussian kernel
        sigma: Standard deviation for Gaussian

    Returns:
        Smoothed attributions
    """
    if attributions.dim() == 4:
        batch, channels, height, width = attributions.shape
        attributions = attributions.reshape(batch, channels * height * width)

    import torch.nn.functional as F

    x = attributions.unsqueeze(0).unsqueeze(0)

    smoothed = F.avg_pool2d(
        x,
        kernel_size=kernel_size,
        stride=1,
        padding=kernel_size // 2,
    )

    smoothed = smoothed.squeeze(0).squeeze(0)

    return smoothed.reshape_as(attributions)


def compute_perturbation_curve(
    model: nn.Module,
    input_tensor: Tensor,
    attributions: Tensor,
    num_steps: int = 10,
    baseline: Optional[Tensor] = None,
) -> Tuple[List[float], List[float]]:
    """Compute insertion/deletion curve for attributions.

    Args:
        model: Model to evaluate
        input_tensor: Input tensor
        attributions: Attribution values
        num_steps: Number of perturbation steps
        baseline: Baseline for perturbation (default: zeros)

    Returns:
        Tuple of (perturbation_percentages, prediction_scores)
    """
    if baseline is None:
        baseline = torch.zeros_like(input_tensor)

    abs_attr = attributions.abs()
    flat_attr = abs_attr.flatten()
    sorted_indices = flat_attr.argsort(descending=True).tolist()

    predictions = []
    percentages = []

    current_input = baseline.clone()

    for step in range(num_steps):
        num_features_to_add = len(sorted_indices) // num_steps * (step + 1)

        for i in sorted_indices[:num_features_to_add]:
            idx = np.unravel_index(i, attributions.shape)
            current_input[idx] = input_tensor[idx]

        with torch.no_grad():
            output = model(current_input)
            pred = torch.softmax(output, dim=-1)[0].max().item()

        predictions.append(pred)
        percentages.append(num_features_to_add / len(sorted_indices) * 100)

    return percentages, predictions


def compute_sparsity_metrics(
    attributions: Tensor,
) -> Dict[str, float]:
    """Compute sparsity-related metrics for attributions.

    Args:
        attributions: Attribution tensor

    Returns:
        Dictionary of sparsity metrics
    """
    flat = attributions.flatten()
    abs_flat = flat.abs()

    threshold = abs_flat.mean()
    sparse_fraction = (abs_flat < threshold).float().mean().item()

    nonzero_fraction = (abs_flat > 1e-8).float().mean().item()

    gini = compute_gini_coefficient(abs_flat.cpu().numpy())

    return {
        "sparsity_ratio": sparse_fraction,
        "nonzero_fraction": nonzero_fraction,
        "gini_coefficient": gini,
    }


def compute_gini_coefficient(values: np.ndarray) -> float:
    """Compute Gini coefficient for measuring attribution inequality."""
    sorted_values = np.sort(np.abs(values))
    n = len(sorted_values)
    cumsum = np.cumsum(sorted_values)
    return (2 * np.sum((np.arange(1, n + 1)) * sorted_values)) / (n * cumsum[-1]) - (
        n + 1
    ) / n


def create_attribution_visualization_data(
    attributions: Tensor,
    input_tensor: Optional[Tensor] = None,
    overlay_alpha: float = 0.5,
) -> Dict:
    """Create data structure for visualization.

    Args:
        attributions: Attribution values
        input_tensor: Original input (optional)
        overlay_alpha: Alpha for overlay

    Returns:
        Dictionary with visualization data
    """
    data = {
        "attributions": attributions.cpu().numpy().tolist(),
        "shape": list(attributions.shape),
    }

    if input_tensor is not None:
        data["input"] = input_tensor.cpu().numpy().tolist()
        data["overlay_alpha"] = overlay_alpha

    stats = get_attribution_stats(attributions)
    data["stats"] = stats

    return data
