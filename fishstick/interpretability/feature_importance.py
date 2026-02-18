"""
Feature Importance Methods

Implements various feature importance techniques:
- Permutation Importance
- SHAP-style explanations
- LIME-style local explanations
- Feature Ablation
"""

from typing import Optional, List, Dict, Union, Tuple, Callable, Any
import torch
from torch import nn, Tensor
import torch.nn.functional as F
from abc import ABC, abstractmethod
import math
import random


class FeatureImportanceBase(ABC):
    """Base class for feature importance methods."""

    def __init__(self, model: nn.Module):
        self.model = model
        self.model.eval()

    @abstractmethod
    def attribute(self, x: Tensor, **kwargs) -> Tensor:
        raise NotImplementedError


class PermutationImportance(FeatureImportanceBase):
    """Permutation-based feature importance.

    Measures importance by permuting features and observing performance drop.

    Args:
        model: PyTorch model
        n_repeats: Number of permutation repeats
        metric: Metric function (default: negative cross-entropy)
    """

    def __init__(
        self, model: nn.Module, n_repeats: int = 5, metric: Optional[Callable] = None
    ):
        super().__init__(model)
        self.n_repeats = n_repeats
        self.metric = metric or self._default_metric

    def _default_metric(self, output: Tensor, target: Tensor) -> Tensor:
        return -F.cross_entropy(output, target, reduction="none")

    def attribute(
        self,
        x: Tensor,
        target: Optional[Union[int, Tensor]] = None,
        feature_dim: int = 1,
        baseline_value: Optional[float] = None,
    ) -> Tensor:
        x = x.clone()

        with torch.no_grad():
            baseline_output = self.model(x)

        if target is None:
            target_indices = baseline_output.argmax(dim=-1)
        elif isinstance(target, int):
            target_indices = torch.full(
                (x.size(0),), target, dtype=torch.long, device=x.device
            )
        else:
            target_indices = target

        baseline_score = self.metric(baseline_output, target_indices)

        n_features = x.size(feature_dim)
        importance = torch.zeros(x.size(0), n_features, device=x.device)

        for feat_idx in range(n_features):
            score_drops = []

            for _ in range(self.n_repeats):
                x_permuted = x.clone()

                perm_indices = torch.randperm(x.size(0), device=x.device)

                permuted_feature = x_permuted[perm_indices, feat_idx].clone()

                if x.dim() == 2:
                    x_permuted[:, feat_idx] = permuted_feature
                elif x.dim() > 2:
                    x_permuted.select(feature_dim, feat_idx).copy_(
                        x.select(feature_dim, feat_idx)[perm_indices]
                    )

                with torch.no_grad():
                    perm_output = self.model(x_permuted)

                perm_score = self.metric(perm_output, target_indices)

                score_drop = baseline_score - perm_score
                score_drops.append(score_drop)

            importance[:, feat_idx] = torch.stack(score_drops).mean(dim=0)

        return importance


class FeatureAblation(FeatureImportanceBase):
    """Feature ablation for importance estimation.

    Measures importance by setting features to a baseline value.

    Args:
        model: PyTorch model
        baseline: Baseline value for ablation (default: 0)
        ablation_type: 'zero' or 'mean' or 'random'
    """

    def __init__(
        self, model: nn.Module, baseline: float = 0.0, ablation_type: str = "zero"
    ):
        super().__init__(model)
        self.baseline = baseline
        self.ablation_type = ablation_type

    def attribute(
        self,
        x: Tensor,
        target: Optional[Union[int, Tensor]] = None,
        feature_dim: int = 1,
    ) -> Tensor:
        x = x.clone()

        with torch.no_grad():
            baseline_output = self.model(x)

        if target is None:
            target_indices = baseline_output.argmax(dim=-1)
        elif isinstance(target, int):
            target_indices = torch.full(
                (x.size(0),), target, dtype=torch.long, device=x.device
            )
        else:
            target_indices = target

        baseline_score = (
            F.softmax(baseline_output, dim=-1)
            .gather(1, target_indices.unsqueeze(1))
            .squeeze(1)
        )

        n_features = x.size(feature_dim)
        importance = torch.zeros(x.size(0), n_features, device=x.device)

        for feat_idx in range(n_features):
            x_ablated = x.clone()

            if self.ablation_type == "zero":
                ablated_value = 0.0
            elif self.ablation_type == "mean":
                ablated_value = x.select(feature_dim, feat_idx).mean().item()
            elif self.ablation_type == "random":
                ablated_value = torch.randn(1).item() * x.std().item()
            else:
                ablated_value = self.baseline

            if x.dim() == 2:
                x_ablated[:, feat_idx] = ablated_value
            elif x.dim() > 2:
                x_ablated.select(feature_dim, feat_idx).fill_(ablated_value)

            with torch.no_grad():
                ablated_output = self.model(x_ablated)

            ablated_score = (
                F.softmax(ablated_output, dim=-1)
                .gather(1, target_indices.unsqueeze(1))
                .squeeze(1)
            )

            importance[:, feat_idx] = baseline_score - ablated_score

        return importance


class SHAPExplainer(FeatureImportanceBase):
    """SHAP-style (Shapley values) feature attribution.

    Approximates Shapley values using sampling.

    Args:
        model: PyTorch model
        n_samples: Number of samples for approximation
        background: Background data for masking
    """

    def __init__(
        self,
        model: nn.Module,
        n_samples: int = 100,
        background: Optional[Tensor] = None,
    ):
        super().__init__(model)
        self.n_samples = n_samples
        self.background = background

    def attribute(
        self,
        x: Tensor,
        target: Optional[Union[int, Tensor]] = None,
        background: Optional[Tensor] = None,
        feature_dim: int = 1,
    ) -> Tensor:
        x = x.clone()

        if background is None:
            background = self.background
        if background is None:
            background = torch.zeros_like(x)

        batch_size = x.size(0)
        n_features = x.size(feature_dim)

        with torch.no_grad():
            output = self.model(x)

        if target is None:
            target_indices = output.argmax(dim=-1)
        elif isinstance(target, int):
            target_indices = torch.full(
                (batch_size,), target, dtype=torch.long, device=x.device
            )
        else:
            target_indices = target

        shap_values = torch.zeros(batch_size, n_features, device=x.device)

        for sample_idx in range(self.n_samples):
            coalition = torch.rand(batch_size, n_features, device=x.device) > 0.5

            for feat_idx in range(n_features):
                coalition_with = coalition.clone()
                coalition_with[:, feat_idx] = True

                coalition_without = coalition.clone()
                coalition_without[:, feat_idx] = False

                x_with = self._apply_coalition(
                    x, background, coalition_with, feature_dim
                )
                x_without = self._apply_coalition(
                    x, background, coalition_without, feature_dim
                )

                with torch.no_grad():
                    out_with = self.model(x_with)
                    out_without = self.model(x_without)

                score_with = (
                    F.softmax(out_with, dim=-1)
                    .gather(1, target_indices.unsqueeze(1))
                    .squeeze(1)
                )
                score_without = (
                    F.softmax(out_without, dim=-1)
                    .gather(1, target_indices.unsqueeze(1))
                    .squeeze(1)
                )

                shap_values[:, feat_idx] += score_with - score_without

        shap_values /= self.n_samples

        return shap_values

    def _apply_coalition(
        self, x: Tensor, background: Tensor, coalition: Tensor, feature_dim: int
    ) -> Tensor:
        if x.dim() == 2:
            coalition_expanded = coalition
        else:
            shape = [x.size(i) for i in range(x.dim())]
            shape[feature_dim] = -1
            for i in range(2, x.dim()):
                coalition = coalition.unsqueeze(-1)
            coalition_expanded = coalition.expand_as(x)

        return torch.where(coalition_expanded.bool(), x, background)


class KernelSHAP(FeatureImportanceBase):
    """Kernel SHAP - efficient SHAP approximation.

    Uses weighted linear regression to estimate Shapley values.

    Args:
        model: PyTorch model
        n_samples: Number of samples
        background: Background data
    """

    def __init__(
        self,
        model: nn.Module,
        n_samples: int = 1000,
        background: Optional[Tensor] = None,
    ):
        super().__init__(model)
        self.n_samples = n_samples
        self.background = background

    def attribute(
        self,
        x: Tensor,
        target: Optional[Union[int, Tensor]] = None,
        background: Optional[Tensor] = None,
        feature_dim: int = 1,
    ) -> Tensor:
        x = x.clone()

        if background is None:
            background = self.background
        if background is None:
            background = torch.zeros_like(x)

        batch_size = x.size(0)
        n_features = x.size(feature_dim)

        with torch.no_grad():
            output = self.model(x)

        if target is None:
            target_indices = output.argmax(dim=-1)
        elif isinstance(target, int):
            target_indices = torch.full(
                (batch_size,), target, dtype=torch.long, device=x.device
            )
        else:
            target_indices = target

        all_shap_values = []

        for b in range(batch_size):
            coalitions = torch.zeros(self.n_samples, n_features, device=x.device)

            for i in range(self.n_samples):
                k = random.randint(1, n_features - 1) if n_features > 1 else 1
                selected = random.sample(range(n_features), k)
                coalitions[i, selected] = 1

            masked_inputs = []
            for i in range(self.n_samples):
                coalition = coalitions[i]
                if x.dim() == 2:
                    masked = torch.where(coalition.bool(), x[b], background[b])
                else:
                    coalition_exp = coalition.view(-1, *([1] * (x.dim() - 1)))
                    masked = torch.where(coalition_exp.bool(), x[b], background[b])
                masked_inputs.append(masked)

            masked_inputs = torch.stack(masked_inputs)

            with torch.no_grad():
                outputs = self.model(masked_inputs)

            scores = F.softmax(outputs, dim=-1)[:, target_indices[b]]

            weights = self._compute_kernel_weights(coalitions, n_features)

            ones = torch.ones(self.n_samples, 1, device=x.device)
            X = torch.cat([ones, coalitions], dim=1)

            W = torch.diag(weights)
            XtW = X.T @ W
            beta = torch.linalg.solve(
                XtW @ X + 1e-6 * torch.eye(n_features + 1, device=x.device),
                XtW @ scores,
            )

            shap_values = beta[1:]
            all_shap_values.append(shap_values)

        return torch.stack(all_shap_values)

    def _compute_kernel_weights(self, coalitions: Tensor, n_features: int) -> Tensor:
        n_coalition_features = coalitions.sum(dim=1)

        weights = torch.zeros_like(n_coalition_features)
        for i, k in enumerate(n_coalition_features):
            k_val = int(k.item())
            if k_val == 0 or k_val == n_features:
                weights[i] = 0
            else:
                weights[i] = (n_features - 1) / (k_val * (n_features - k_val))

        return weights


class LIMEExplainer(FeatureImportanceBase):
    """LIME-style local interpretable model explanations.

    Fits a local linear model around the prediction.

    Args:
        model: PyTorch model
        n_samples: Number of perturbation samples
        kernel_width: Width of the locality kernel
    """

    def __init__(
        self, model: nn.Module, n_samples: int = 1000, kernel_width: float = 0.25
    ):
        super().__init__(model)
        self.n_samples = n_samples
        self.kernel_width = kernel_width

    def attribute(
        self,
        x: Tensor,
        target: Optional[Union[int, Tensor]] = None,
        feature_dim: int = 1,
        perturbation_std: float = 0.1,
    ) -> Tensor:
        x = x.clone()
        batch_size = x.size(0)
        n_features = x.size(feature_dim)

        with torch.no_grad():
            output = self.model(x)

        if target is None:
            target_indices = output.argmax(dim=-1)
        elif isinstance(target, int):
            target_indices = torch.full(
                (batch_size,), target, dtype=torch.long, device=x.device
            )
        else:
            target_indices = target

        all_importance = []

        for b in range(batch_size):
            perturbations = (
                torch.randn(self.n_samples, n_features, device=x.device)
                * perturbation_std
            )
            perturbed_x = x[b].unsqueeze(0) + perturbations

            with torch.no_grad():
                perturbed_output = self.model(perturbed_x)

            scores = F.softmax(perturbed_output, dim=-1)[:, target_indices[b]]

            distances = (perturbations**2).sum(dim=1)
            weights = torch.exp(-distances / (2 * self.kernel_width**2))

            W = torch.diag(weights)
            XtW = perturbations.T @ W
            beta = torch.linalg.solve(
                XtW @ perturbations + 1e-6 * torch.eye(n_features, device=x.device),
                XtW @ scores,
            )

            all_importance.append(beta.abs())

        return torch.stack(all_importance)


class TreeSHAP(FeatureImportanceBase):
    """Tree SHAP approximation for neural networks.

    Uses tree-like partitioning for efficient computation.

    Args:
        model: PyTorch model
        n_samples: Number of samples
    """

    def __init__(self, model: nn.Module, n_samples: int = 100):
        super().__init__(model)
        self.n_samples = n_samples

    def attribute(
        self,
        x: Tensor,
        target: Optional[Union[int, Tensor]] = None,
        feature_dim: int = 1,
    ) -> Tensor:
        return SHAPExplainer(self.model, n_samples=self.n_samples).attribute(
            x, target, feature_dim=feature_dim
        )


class OcclusionSensitivity(FeatureImportanceBase):
    """Occlusion sensitivity analysis.

    Measures importance by occluding regions of the input.

    Args:
        model: PyTorch model
        window_size: Size of occlusion window
        stride: Stride for sliding window
    """

    def __init__(self, model: nn.Module, window_size: int = 3, stride: int = 1):
        super().__init__(model)
        self.window_size = window_size
        self.stride = stride

    def attribute(
        self,
        x: Tensor,
        target: Optional[Union[int, Tensor]] = None,
        occlusion_value: float = 0.0,
    ) -> Tensor:
        x = x.clone()

        with torch.no_grad():
            baseline_output = self.model(x)

        if target is None:
            target_indices = baseline_output.argmax(dim=-1)
        elif isinstance(target, int):
            target_indices = torch.full(
                (x.size(0),), target, dtype=torch.long, device=x.device
            )
        else:
            target_indices = target

        baseline_score = (
            F.softmax(baseline_output, dim=-1)
            .gather(1, target_indices.unsqueeze(1))
            .squeeze(1)
        )

        if x.dim() == 2:
            return self._occlude_1d(x, target_indices, baseline_score, occlusion_value)
        elif x.dim() == 4:
            return self._occlude_2d(x, target_indices, baseline_score, occlusion_value)
        else:
            raise ValueError(f"Unsupported input dimension: {x.dim()}")

    def _occlude_1d(
        self, x: Tensor, target: Tensor, baseline_score: Tensor, occlusion_value: float
    ) -> Tensor:
        batch_size, n_features = x.shape
        sensitivity = torch.zeros_like(x)

        for i in range(0, n_features - self.window_size + 1, self.stride):
            x_occluded = x.clone()
            end = min(i + self.window_size, n_features)
            x_occluded[:, i:end] = occlusion_value

            with torch.no_grad():
                output = self.model(x_occluded)

            score = F.softmax(output, dim=-1).gather(1, target.unsqueeze(1)).squeeze(1)
            sensitivity[:, i:end] += (baseline_score - score).unsqueeze(-1)

        return sensitivity

    def _occlude_2d(
        self, x: Tensor, target: Tensor, baseline_score: Tensor, occlusion_value: float
    ) -> Tensor:
        batch_size, channels, height, width = x.shape
        sensitivity = torch.zeros(batch_size, 1, height, width, device=x.device)
        counts = torch.zeros(height, width, device=x.device)

        for i in range(0, height - self.window_size + 1, self.stride):
            for j in range(0, width - self.window_size + 1, self.stride):
                x_occluded = x.clone()
                i_end = min(i + self.window_size, height)
                j_end = min(j + self.window_size, width)
                x_occluded[:, :, i:i_end, j:j_end] = occlusion_value

                with torch.no_grad():
                    output = self.model(x_occluded)

                score = (
                    F.softmax(output, dim=-1).gather(1, target.unsqueeze(1)).squeeze(1)
                )
                sensitivity[:, :, i:i_end, j:j_end] += (baseline_score - score).view(
                    -1, 1, 1, 1
                )
                counts[i:i_end, j:j_end] += 1

        counts = counts.clamp(min=1)
        sensitivity = sensitivity / counts.view(1, 1, height, width)

        return sensitivity.expand(-1, channels, -1, -1)


class FeatureInteraction(FeatureImportanceBase):
    """Feature interaction detection.

    Detects interactions between features.

    Args:
        model: PyTorch model
    """

    def __init__(self, model: nn.Module):
        super().__init__(model)

    def attribute(
        self, x: Tensor, target: Optional[Union[int, Tensor]] = None, max_order: int = 2
    ) -> Dict[str, Tensor]:
        x = x.clone()

        if x.dim() != 2:
            raise ValueError("FeatureInteraction only supports 2D inputs")

        n_features = x.size(1)

        main_effects = FeatureAblation(self.model).attribute(x, target)

        interactions = {}
        interactions["main"] = main_effects

        if max_order >= 2:
            pairwise = torch.zeros(x.size(0), n_features, n_features, device=x.device)

            with torch.no_grad():
                baseline_output = self.model(x)

            if target is None:
                target_indices = baseline_output.argmax(dim=-1)
            elif isinstance(target, int):
                target_indices = torch.full(
                    (x.size(0),), target, dtype=torch.long, device=x.device
                )
            else:
                target_indices = target

            baseline_score = (
                F.softmax(baseline_output, dim=-1)
                .gather(1, target_indices.unsqueeze(1))
                .squeeze(1)
            )

            for i in range(n_features):
                for j in range(i + 1, n_features):
                    x_ablated_ij = x.clone()
                    x_ablated_ij[:, i] = 0
                    x_ablated_ij[:, j] = 0

                    x_ablated_i = x.clone()
                    x_ablated_i[:, i] = 0

                    x_ablated_j = x.clone()
                    x_ablated_j[:, j] = 0

                    with torch.no_grad():
                        out_ij = self.model(x_ablated_ij)
                        out_i = self.model(x_ablated_i)
                        out_j = self.model(x_ablated_j)

                    score_ij = (
                        F.softmax(out_ij, dim=-1)
                        .gather(1, target_indices.unsqueeze(1))
                        .squeeze(1)
                    )
                    score_i = (
                        F.softmax(out_i, dim=-1)
                        .gather(1, target_indices.unsqueeze(1))
                        .squeeze(1)
                    )
                    score_j = (
                        F.softmax(out_j, dim=-1)
                        .gather(1, target_indices.unsqueeze(1))
                        .squeeze(1)
                    )

                    interaction_strength = baseline_score - score_i - score_j + score_ij

                    pairwise[:, i, j] = interaction_strength
                    pairwise[:, j, i] = interaction_strength

            interactions["pairwise"] = pairwise

        return interactions


def create_feature_importance(
    method: str, model: nn.Module, **kwargs
) -> FeatureImportanceBase:
    """Factory function to create feature importance methods.

    Args:
        method: Method name ('permutation', 'ablation', 'shap',
                'kernelshap', 'lime', 'occlusion', 'interaction')
        model: PyTorch model
        **kwargs: Additional arguments

    Returns:
        Feature importance method instance
    """
    methods = {
        "permutation": PermutationImportance,
        "ablation": FeatureAblation,
        "shap": SHAPExplainer,
        "kernelshap": KernelSHAP,
        "lime": LIMEExplainer,
        "treeshap": TreeSHAP,
        "occlusion": OcclusionSensitivity,
        "interaction": FeatureInteraction,
    }

    method_lower = method.lower()
    if method_lower not in methods:
        raise ValueError(f"Unknown method: {method}. Available: {list(methods.keys())}")

    return methods[method_lower](model, **kwargs)
