"""
XAI Metrics

Comprehensive metrics for evaluating explainability methods:
- Fidelity metrics (AUC, insertion/deletion)
- Complexity metrics
- Stability metrics
- Correlation metrics
"""

from __future__ import annotations

from typing import Optional, List, Dict, Tuple, Callable
import warnings

import torch
from torch import Tensor, nn
import numpy as np


class FidelityMetrics:
    """Metrics measuring fidelity - how well explanations match model behavior.

    Includes:
    - AUC-ROC for insertion/deletion games
    - Average Precision
    - Correlation-based metrics
    """

    @staticmethod
    def compute_auc_score(
        model: nn.Module,
        input_tensor: Tensor,
        attributions: Tensor,
        baseline: Optional[Tensor] = None,
        target_class: Optional[int] = None,
        num_steps: int = 100,
    ) -> Dict[str, float]:
        """Compute AUC scores for insertion and deletion curves.

        Args:
            model: Model to evaluate
            input_tensor: Original input
            attributions: Attribution values
            baseline: Baseline for perturbation
            target_class: Target class for evaluation
            num_steps: Number of perturbation steps

        Returns:
            Dictionary with insertion_auc, deletion_auc
        """
        if baseline is None:
            baseline = torch.zeros_like(input_tensor)

        if target_class is None:
            model.eval()
            with torch.no_grad():
                output = model(input_tensor)
                target_class = output.argmax(dim=-1).item()

        insertion_curve = FidelityMetrics._compute_insertion_curve(
            model, input_tensor, attributions, baseline, target_class, num_steps
        )

        deletion_curve = FidelityMetrics._compute_deletion_curve(
            model, input_tensor, attributions, baseline, target_class, num_steps
        )

        insertion_auc = FidelityMetrics._compute_auc(insertion_curve)
        deletion_auc = FidelityMetrics._compute_auc(deletion_curve)

        return {
            "insertion_auc": insertion_auc,
            "deletion_auc": deletion_auc,
        }

    @staticmethod
    def _compute_insertion_curve(
        model: nn.Module,
        input_tensor: Tensor,
        attributions: Tensor,
        baseline: Tensor,
        target_class: int,
        num_steps: int,
    ) -> List[float]:
        """Compute insertion curve - gradually adding important features."""
        abs_attr = attributions.abs().flatten()
        sorted_indices = abs_attr.argsort(descending=True).tolist()

        current = baseline.clone()
        scores = []

        step_size = max(1, len(sorted_indices) // num_steps)

        for step in range(num_steps):
            num_features = step * step_size

            for i in sorted_indices[:num_features]:
                idx = np.unravel_index(i, attributions.shape)
                current[idx] = input_tensor[idx]

            with torch.no_grad():
                output = model(current)
                score = torch.softmax(output, dim=-1)[0, target_class].item()

            scores.append(score)

        return scores

    @staticmethod
    def _compute_deletion_curve(
        model: nn.Module,
        input_tensor: Tensor,
        attributions: Tensor,
        baseline: Tensor,
        target_class: int,
        num_steps: int,
    ) -> List[float]:
        """Compute deletion curve - gradually removing important features."""
        abs_attr = attributions.abs().flatten()
        sorted_indices = abs_attr.argsort(descending=True).tolist()

        current = input_tensor.clone()
        scores = []

        step_size = max(1, len(sorted_indices) // num_steps)

        for step in range(num_steps):
            num_features = step * step_size

            for i in sorted_indices[:num_features]:
                idx = np.unravel_index(i, attributions.shape)
                current[idx] = baseline[idx]

            with torch.no_grad():
                output = model(current)
                score = torch.softmax(output, dim=-1)[0, target_class].item()

            scores.append(score)

        return scores

    @staticmethod
    def _compute_auc(curve: List[float]) -> float:
        """Compute area under curve."""
        if not curve:
            return 0.0

        curve_array = np.array(curve)
        x = np.linspace(0, 1, len(curve_array))

        auc = np.trapz(curve_array, x)
        return float(auc)

    @staticmethod
    def compute_average_precision(
        attributions: Tensor,
        ground_truth_mask: Optional[Tensor] = None,
    ) -> float:
        """Compute average precision from attributions.

        Args:
            attributions: Attribution values
            ground_truth_mask: Binary mask of important features

        Returns:
            Average precision score
        """
        if ground_truth_mask is None:
            warnings.warn("No ground truth provided, returning 0.0")
            return 0.0

        pred_scores = attributions.abs().flatten()
        ground_truth = ground_truth_mask.flatten()

        from sklearn.metrics import average_precision_score

        ap = average_precision_score(
            ground_truth.cpu().numpy(), pred_scores.cpu().numpy()
        )

        return float(ap)


class ComplexityMetrics:
    """Metrics measuring explanation complexity.

    Includes:
    - Sparsity
    - Number of salient features
    - Compactness
    """

    @staticmethod
    def compute_sparsity(attributions: Tensor, threshold: float = 0.01) -> float:
        """Compute sparsity score (fraction of features below threshold).

        Args:
            attributions: Attribution values
            threshold: Threshold for considering a feature important

        Returns:
            Sparsity score between 0 and 1
        """
        abs_attr = attributions.abs()
        threshold_tensor = torch.tensor(threshold, device=attr.device)
        sparse_count = (abs_attr < threshold_tensor).float().mean().item()
        return float(sparse_count)

    @staticmethod
    def compute_number_of_features(
        attributions: Tensor,
        threshold: Union[float, str] = "auto",
    ) -> int:
        """Count number of important features.

        Args:
            attributions: Attribution values
            threshold: Threshold or 'auto'

        Returns:
            Number of important features
        """
        abs_attr = attributions.abs()

        if isinstance(threshold, str) and threshold == "auto":
            threshold = abs_attr.mean().item() * 0.1

        important_count = (abs_attr > threshold).sum().item()
        return int(important_count)

    @staticmethod
    def compute_compactness(
        attributions: Tensor,
        method: str = "entropy",
    ) -> float:
        """Compute compactness of explanation.

        Args:
            attributions: Attribution values
            method: Method for computing compactness

        Returns:
            Compactness score
        """
        abs_attr = attributions.abs()
        flat = abs_attr.flatten()
        flat_normalized = flat / (flat.sum() + 1e-8)

        if method == "entropy":
            entropy = -(flat_normalized * torch.log(flat_normalized + 1e-8)).sum()
            max_entropy = torch.log(torch.tensor(flat.numel()))
            compactness = 1 - (entropy / max_entropy)
            return compactness.item()

        elif method == "gini":
            sorted_values = torch.sort(flat.abs())[0]
            n = len(sorted_values)
            cumsum = torch.cumsum(sorted_values, dim=0)
            gini = (2 * torch.sum((torch.arange(1, n + 1).float()) * sorted_values)) / (
                n * cumsum[-1]
            ) - (n + 1) / n
            return 1 - gini.item()

        return 0.0


class StabilityMetrics:
    """Metrics measuring explanation stability.

    Includes:
    - Sensitivity to noise
    - Robustness under perturbations
    - Consistency across similar inputs
    """

    @staticmethod
    def compute_sensitivity(
        model: nn.Module,
        input_tensor: Tensor,
        attribution_fn: Callable,
        noise_level: float = 0.1,
        num_samples: int = 10,
    ) -> float:
        """Compute sensitivity - variation in attributions under noise.

        Args:
            model: Model to evaluate
            input_tensor: Original input
            attribution_fn: Function to compute attributions
            noise_level: Standard deviation of noise
            num_samples: Number of noisy samples

        Returns:
            Sensitivity score (lower is better)
        """
        original_attr = attribution_fn(input_tensor)

        all_attrs = [original_attr]

        for _ in range(num_samples):
            noise = torch.randn_like(input_tensor) * noise_level
            noisy_input = input_tensor + noise

            noisy_attr = attribution_fn(noisy_input)
            all_attrs.append(noisy_attr)

        stacked = torch.stack(all_attrs)
        std = stacked.std(dim=0)

        sensitivity = std.mean().item()
        return float(sensitivity)

    @staticmethod
    def compute_robustness(
        model: nn.Module,
        input_tensor: Tensor,
        attributions: Tensor,
        perturbations: List[Tensor],
    ) -> float:
        """Compute robustness of attributions under perturbations.

        Args:
            model: Model to evaluate
            input_tensor: Original input
            attributions: Original attributions
            perturbations: List of perturbed inputs

        Returns:
            Robustness score (higher is better)
        """
        attr_magnitude = attributions.abs().mean().item()

        robustness_scores = []

        for perturbed in perturbations:
            model.eval()
            with torch.no_grad():
                orig_output = model(input_tensor)
                pert_output = model(perturbed)

                orig_pred = torch.softmax(orig_output, dim=-1)
                pert_pred = torch.softmax(pert_output, dim=-1)

                pred_diff = (orig_pred - pert_pred).abs().mean().item()

                robustness_scores.append(1 - pred_diff)

        if not robustness_scores:
            return 0.0

        return float(np.mean(robustness_scores))

    @staticmethod
    def compute_consistency(
        attribution_fn: Callable,
        similar_inputs: List[Tensor],
    ) -> float:
        """Compute consistency across similar inputs.

        Args:
            attribution_fn: Function to compute attributions
            similar_inputs: List of similar input tensors

        Returns:
            Consistency score (higher is better)
        """
        attributions = [attr_fn(inp) for inp in similar_inputs]

        if len(attributions) < 2:
            return 1.0

        stacked = torch.stack(attributions)

        pairwise_corrs = []

        for i in range(len(attributions)):
            for j in range(i + 1, len(attributions)):
                corr = torch.corrcoef(
                    torch.stack([stacked[i].flatten(), stacked[j].flatten()])
                )[0, 1].item()
                pairwise_corrs.append(corr)

        return float(np.mean(pairwise_corrs))


class CorrelationMetrics:
    """Metrics based on correlation with model predictions.

    Includes:
    - Pearson correlation
    - Spearman correlation
    - Model agreement
    """

    @staticmethod
    def compute_pearson_correlation(
        attributions: Tensor,
        predictions: Tensor,
    ) -> float:
        """Compute Pearson correlation between attributions and predictions.

        Args:
            attributions: Attribution values
            predictions: Model predictions

        Returns:
            Pearson correlation coefficient
        """
        attr_flat = attributions.flatten()
        pred_flat = predictions.flatten()

        attr_mean = attr_flat.mean()
        pred_mean = pred_flat.mean()

        covariance = ((attr_flat - attr_mean) * (pred_flat - pred_mean)).mean()
        attr_std = attr_flat.std()
        pred_std = pred_flat.std()

        if attr_std == 0 or pred_std == 0:
            return 0.0

        correlation = covariance / (attr_std * pred_std)
        return float(correlation.item())

    @staticmethod
    def compute_spearman_correlation(
        attributions: Tensor,
        predictions: Tensor,
    ) -> float:
        """Compute Spearman rank correlation.

        Args:
            attributions: Attribution values
            predictions: Model predictions

        Returns:
            Spearman correlation coefficient
        """
        attr_np = attributions.flatten().cpu().numpy()
        pred_np = predictions.flatten().cpu().numpy()

        from scipy.stats import spearmanr

        corr, _ = spearmanr(attr_np, pred_np)

        return float(corr)

    @staticmethod
    def compute_model_agreement(
        model: nn.Module,
        input_tensor: Tensor,
        attributions: Tensor,
        target_class: Optional[int] = None,
    ) -> float:
        """Compute agreement between attribution sign and prediction change.

        Args:
            model: Model to evaluate
            input_tensor: Original input
            attributions: Attribution values
            target_class: Target class

        Returns:
            Agreement score
        """
        if target_class is None:
            with torch.no_grad():
                output = model(input_tensor)
                target_class = output.argmax(dim=-1).item()

        baseline = torch.zeros_like(input_tensor)

        positive_mask = (attributions > 0).float()
        negative_mask = (attributions < 0).float()

        pos_perturbed = input_tensor * positive_mask + baseline * negative_mask
        neg_perturbed = input_tensor * negative_mask + baseline * positive_mask

        model.eval()
        with torch.no_grad():
            orig_output = model(input_tensor)
            orig_prob = torch.softmax(orig_output, dim=-1)[0, target_class].item()

            pos_output = model(pos_perturbed)
            pos_prob = torch.softmax(pos_output, dim=-1)[0, target_class].item()

            neg_output = model(neg_perturbed)
            neg_prob = torch.softmax(neg_output, dim=-1)[0, target_class].item()

        agreement = 0.0
        if pos_prob > orig_prob:
            agreement += 0.5
        if neg_prob < orig_prob:
            agreement += 0.5

        return agreement


class ExplanationMetrics:
    """Unified class for comprehensive explanation evaluation."""

    def __init__(self, model: nn.Module):
        self.model = model
        self.model.eval()

    def evaluate(
        self,
        input_tensor: Tensor,
        attributions: Tensor,
        baseline: Optional[Tensor] = None,
        target_class: Optional[int] = None,
        ground_truth_mask: Optional[Tensor] = None,
        num_samples: int = 10,
    ) -> Dict[str, float]:
        """Compute all metrics for an explanation.

        Args:
            input_tensor: Input to evaluate
            attributions: Attribution values
            baseline: Baseline for perturbation
            target_class: Target class
            ground_truth_mask: Ground truth important features
            num_samples: Number of samples for stability

        Returns:
            Dictionary with all metric scores
        """
        if target_class is None:
            with torch.no_grad():
                output = self.model(input_tensor)
                target_class = output.argmax(dim=-1).item()

        metrics = {}

        auc_metrics = FidelityMetrics.compute_auc_score(
            self.model, input_tensor, attributions, baseline, target_class
        )
        metrics.update(auc_metrics)

        if ground_truth_mask is not None:
            ap = FidelityMetrics.compute_average_precision(
                attributions, ground_truth_mask
            )
            metrics["average_precision"] = ap

        metrics["sparsity"] = ComplexityMetrics.compute_sparsity(attributions)
        metrics["num_features"] = ComplexityMetrics.compute_number_of_features(
            attributions
        )
        metrics["compactness"] = ComplexityMetrics.compute_compactness(attributions)

        return metrics


def create_metric_evaluator(
    model: nn.Module,
) -> ExplanationMetrics:
    """Factory function to create metric evaluator.

    Args:
        model: Model to evaluate

    Returns:
        ExplanationMetrics instance
    """
    return ExplanationMetrics(model)
