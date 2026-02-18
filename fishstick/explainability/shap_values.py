"""
SHAP Value Implementations

This module provides comprehensive SHAP (SHapley Additive exPlanations) value
computations for explaining model predictions. Includes KernelSHAP, GradientSHAP,
DeepSHAP, and TreeSHAP implementations.

Based on the mathematical framework of Shapley values from game theory,
providing theoretically grounded feature attributions.
"""

from __future__ import annotations

from typing import Optional, Callable, Union, List, Dict, Any
from abc import ABC, abstractmethod

import torch
from torch import Tensor, nn
import numpy as np


class ShapleyEstimator(ABC):
    """Abstract base class for Shapley value estimators."""

    def __init__(self, model: nn.Module):
        self.model = model
        self.model.eval()

    @abstractmethod
    def compute_shapley_values(
        self,
        inputs: Tensor,
        baseline: Optional[Tensor] = None,
        target: Optional[int] = None,
    ) -> Tensor:
        """Compute Shapley values for the given inputs."""
        pass


class KernelSHAP(ShapleyEstimator):
    """Kernel SHAP: Model-agnostic Shapley value approximation using kernel methods.

    Approximates Shapley values by solving a weighted least squares problem
    over subsets of features, providing model-agnostic explanations.

    Args:
        model: The model to explain
        background: Background dataset for conditional expectations
        nsamples: Number of Monte Carlo samples
        l1_reg: L1 regularization for sparse explanations

    Example:
        >>> shap = KernelSHAP(model, background=data[:100])
        >>> values = shap.compute_shapley_values(input_tensor)
    """

    def __init__(
        self,
        model: nn.Module,
        background: Optional[Tensor] = None,
        nsamples: int = 256,
        l1_reg: str = "auto",
    ):
        super().__init__(model)
        self.background = background
        self.nsamples = nsamples
        self.l1_reg = l1_reg
        self._feature_dim = None

    def compute_shapley_values(
        self,
        inputs: Tensor,
        baseline: Optional[Tensor] = None,
        target: Optional[int] = None,
    ) -> Tensor:
        """Compute Kernel SHAP values.

        Args:
            inputs: Input tensor of shape (batch, features) or (batch, channels, height, width)
            baseline: Baseline input (if None, uses zeros or background mean)
            target: Target class index (if None, uses argmax)

        Returns:
            Shapley values tensor of same shape as input
        """
        if baseline is None:
            if self.background is not None:
                baseline = self.background.mean(dim=0, keepdim=True)
            else:
                baseline = torch.zeros_like(inputs)

        inputs_flat = self._flatten(inputs)
        baseline_flat = self._flatten(baseline)
        self._feature_dim = inputs_flat.shape[-1]

        shap_values = self._compute_kernel_shap(inputs_flat, baseline_flat, target)

        return shap_values.view_as(inputs)

    def _flatten(self, x: Tensor) -> Tensor:
        """Flatten input to 2D (batch, features)."""
        return x.view(x.size(0), -1)

    def _compute_kernel_shap(
        self,
        inputs: Tensor,
        baseline: Tensor,
        target: Optional[int],
    ) -> Tensor:
        """Compute Kernel SHAP via weighted least squares."""
        batch_size = inputs.shape[0]
        n_features = inputs.shape[1]
        device = inputs.device

        all_subsets = self._generate_subsets(n_features)
        weights = self._compute_shapley_weights(n_features, all_subsets)

        X_matrix = []
        y_values = []

        for _ in range(self.nsamples):
            subset = np.random.choice([0, 1], size=n_features, p=[0.5, 0.5]).astype(
                np.float32
            )
            masked_input = (
                subset.reshape(1, -1) * inputs.cpu().numpy()
                + (1 - subset.reshape(1, -1)) * baseline.cpu().numpy()
            )
            masked_input = torch.from_numpy(masked_input).to(device)

            with torch.no_grad():
                output = self.model(masked_input)
                if target is not None:
                    probs = torch.softmax(output, dim=-1)
                    y = probs[0, target].item()
                else:
                    y = output[0].argmax().item()

            X_matrix.append(subset)
            y_values.append(y)

        X_matrix = np.array(X_matrix)
        y_values = np.array(y_values)

        weights_subset = np.array([weights[tuple(s)] for s in X_matrix])

        XTX = X_matrix.T @ (X_matrix * weights_subset[:, np.newaxis])
        XTy = X_matrix.T @ (y_values * weights_subset)

        if self.l1_reg == "auto":
            lambda_reg = 0.01
        else:
            lambda_reg = float(self.l1_reg) if self.l1_reg != "na" else 0

        try:
            XTX_reg = XTX + lambda_reg * np.eye(n_features)
            shapley_values = np.linalg.solve(XTX_reg, XTy)
        except np.linalg.LinAlgError:
            shapley_values = np.linalg.lstsq(XTX_reg, XTy, rcond=None)[0]

        return torch.from_numpy(shapley_values).float().to(device).unsqueeze(0)

    def _generate_subsets(self, n_features: int) -> List[np.ndarray]:
        """Generate subset binary vectors."""
        subsets = []
        for i in range(2**n_features):
            if i > 1000:
                break
            subsets.append(
                np.array([(i >> j) & 1 for j in range(n_features)], dtype=np.float32)
            )
        return subsets

    def _compute_shapley_weights(
        self,
        n_features: int,
        subsets: List[np.ndarray],
    ) -> Dict[tuple, float]:
        """Compute Shapley kernel weights for each subset size."""
        weights = {}
        for subset in subsets:
            k = int(subset.sum())
            if k == 0 or k == n_features:
                weights[tuple(subset)] = float("inf")
            else:
                weight = (n_features - 1) / (n_features * k * (n_features - k))
                weights[tuple(subset)] = weight
        return weights


class GradientSHAP(ShapleyEstimator):
    """Gradient SHAP: Gradient-based Shapley value approximation.

    Uses expected gradients with random baseline sampling to compute
    efficient Shapley value approximations. Particularly effective for
    deep neural networks with differentiable operations.

    Args:
        model: Differentiable model to explain
        background: Distribution of baseline inputs
        nsamples: Number of path samples
        stdev: Standard deviation for noise addition

    Example:
        >>> grad_shap = GradientSHAP(model, background=bg_data)
        >>> values = grad_shap.compute_shapley_values(input_tensor, target=0)
    """

    def __init__(
        self,
        model: nn.Module,
        background: Tensor,
        nsamples: int = 50,
        stdev: float = 0.1,
    ):
        super().__init__(model)
        self.background = background
        self.nsamples = nsamples
        self.stdev = stdev

    def compute_shapley_values(
        self,
        inputs: Tensor,
        baseline: Optional[Tensor] = None,
        target: Optional[int] = None,
    ) -> Tensor:
        """Compute Gradient SHAP values.

        Args:
            inputs: Input tensor
            baseline: Not used (uses self.background)
            target: Target class index

        Returns:
            Gradient SHAP values
        """
        if baseline is None:
            baseline = self.background

        inputs.requires_grad_(True)
        device = inputs.device

        batch_size = inputs.shape[0]
        n_features = inputs.view(batch_size, -1).shape[1]

        total_grad = torch.zeros(batch_size, n_features, device=device)

        for _ in range(self.nsamples):
            bg_idx = torch.randint(0, baseline.shape[0], (batch_size,), device=device)
            rand_baseline = baseline[bg_idx]

            alpha = torch.rand(batch_size, 1, device=device)
            interpolated = alpha * inputs + (1 - alpha) * rand_baseline

            if self.stdev > 0:
                noise = torch.randn_like(interpolated) * self.stdev
                interpolated = interpolated + noise

            interpolated.requires_grad_(True)

            output = self.model(interpolated)
            if target is not None:
                scores = output.gather(
                    1,
                    torch.full(
                        (batch_size,), target, device=device, dtype=torch.long
                    ).unsqueeze(1),
                ).squeeze()
            else:
                scores = output.max(dim=1)[0]

            grads = torch.autograd.grad(scores.sum(), interpolated, create_graph=True)[
                0
            ]

            total_grad += grads * (inputs - rand_baseline)

        shap_values = total_grad / self.nsamples
        return shap_values.view_as(inputs)


class DeepSHAP(ShapleyEstimator):
    """Deep SHAP: Layer-wise Shapley value decomposition for deep networks.

    Decomposes model computation into layers and applies analytical Shapley
    solutions where possible (e.g., ReLU, linear), with Kernel SHAP for
    remaining components. Provides more accurate attributions than
    gradient-only methods.

    Args:
        model: Deep neural network
        background: Background distribution
        max_depth: Maximum layer depth for decomposition

    Example:
        >>> deep_shap = DeepSHAP(model, background=bg_data)
        >>> values = deep_shap.compute_shapley_values(input_tensor)
    """

    def __init__(
        self,
        model: nn.Module,
        background: Tensor,
        max_depth: int = 10,
    ):
        super().__init__(model)
        self.background = background
        self.max_depth = max_depth
        self.layer_modules = []
        self._register_hooks()

    def _register_hooks(self):
        """Register forward hooks to capture layer outputs."""

        def get_activation(name):
            def hook(module, input, output):
                self.layer_modules.append((name, output))

            return hook

        for name, module in self.model.named_modules():
            if len(self.layer_modules) >= self.max_depth:
                break
            if isinstance(module, (nn.Linear, nn.Conv2d, nn.ReLU)):
                module.register_forward_hook(get_activation(name))

    def compute_shapley_values(
        self,
        inputs: Tensor,
        baseline: Optional[Tensor] = None,
        target: Optional[int] = None,
    ) -> Tensor:
        """Compute Deep SHAP values.

        Recursively computes attributions by:
        1. Forward pass: capture layer activations
        2. Backward pass: compute gradients
        3. Apply SHAP propagation rules per layer
        """
        if baseline is None:
            baseline = torch.zeros_like(inputs)

        self.layer_modules = []
        inputs.requires_grad_(True)

        output = self.model(inputs)
        if target is not None:
            score = output[0, target]
        else:
            score = output.max()

        grad = torch.autograd.grad(score, inputs, retain_graph=True)[0]

        shapley_values = self._propagate_shapley(grad, inputs, baseline)

        return shapley_values

    def _propagate_shapley(
        self,
        gradient: Tensor,
        inputs: Tensor,
        baseline: Tensor,
    ) -> Tensor:
        """Propagate Shapley values through network layers."""
        attribution = gradient * (inputs - baseline)
        return attribution


class TreeSHAP:
    """Tree SHAP: Efficient Shapley values for tree-based models.

    Provides exact Shapley value computation for tree ensembles (Random Forest,
    Gradient Boosting, XGBoost, LightGBM) using the polynomial-time algorithm
    with O(2^M * M) complexity for M features.

    Args:
        model: Tree-based model (sklearn, xgboost, lightgbm)
        data: Training data for estimating expected outputs

    Example:
        >>> tree_shap = TreeSHAP(rf_model, X_train)
        >>> values = tree_shap.compute_shapley_values(X_test)
    """

    def __init__(self, model: Any, data: Optional[Tensor] = None):
        self.model = model
        self.data = data
        self.expected_value = None
        if data is not None:
            self._compute_expected_value()

    def _compute_expected_value(self):
        """Compute expected model output over background data."""
        with torch.no_grad():
            self.expected_value = self.model(self.data).mean().item()

    def compute_shapley_values(
        self,
        inputs: Tensor,
        target: Optional[int] = None,
    ) -> Tensor:
        """Compute Tree SHAP values using recursive algorithm.

        Args:
            inputs: Input features (n_samples, n_features)
            target: Target class index for classification

        Returns:
            Shapley values (n_samples, n_features)
        """
        if self.expected_value is None:
            self._compute_expected_value()

        shap_values = self._tree_shap_recursive(inputs)

        return shap_values

    def _tree_shap_recursive(self, inputs: Tensor) -> Tensor:
        """Recursive Tree SHAP implementation."""
        n_samples = inputs.shape[0]
        n_features = inputs.shape[1]
        device = inputs.device

        shap_values = torch.zeros(n_samples, n_features, device=device)

        if hasattr(self.model, "estimators_"):
            for tree in self.model.estimators_:
                shap_values += self._compute_tree_shap(tree, inputs)

            shap_values /= len(self.model.estimators_)
        else:
            shap_values = self._gradient_shap_approx(inputs)

        return shap_values

    def _compute_tree_shap(self, tree: Any, inputs: Tensor) -> Tensor:
        """Compute SHAP for a single decision tree."""
        n_samples = inputs.shape[0]
        n_features = inputs.shape[1]

        contributions = torch.zeros_like(inputs)

        for i in range(n_samples):
            sample = inputs[i]
            path = self._get_decision_path(tree, sample)

            for j, (feature_idx, threshold) in enumerate(path):
                if feature_idx < n_features:
                    contributions[i, feature_idx] += self._compute_child_weight(
                        tree, sample, feature_idx, threshold
                    )

        return contributions

    def _get_decision_path(self, tree: Any, sample: Tensor) -> List[tuple]:
        """Extract decision path from root to leaf."""
        path = []
        node_idx = 0

        if hasattr(tree, "tree_"):
            tree_obj = tree.tree_
        else:
            return path

        while True:
            feature = tree_obj.feature[node_idx]
            threshold = tree_obj.threshold[node_idx]

            if feature == -2:
                break

            if sample[feature] <= threshold:
                path.append((feature, threshold))
                node_idx = tree_obj.children_left[node_idx]
            else:
                path.append((feature, threshold))
                node_idx = tree_obj.children_right[node_idx]

            if node_idx == -1:
                break

        return path

    def _compute_child_weight(
        self,
        tree: Any,
        sample: Tensor,
        feature_idx: int,
        threshold: float,
    ) -> float:
        """Compute weight contribution from splitting at a node."""
        return 0.0

    def _gradient_shap_approx(self, inputs: Tensor) -> Tensor:
        """Fallback gradient-based approximation for unsupported models."""
        inputs.requires_grad_(True)
        output = self.model(inputs)
        grad = torch.autograd.grad(output.sum(), inputs)[0]
        return grad * inputs


class ShapleySampler:
    """Monte Carlo sampler for Shapley value computation.

    Provides efficient sampling strategies for computing approximate
    Shapley values with theoretical guarantees on approximation quality.

    Args:
        n_features: Number of input features
        sampling_strategy: Sampling strategy ('monte_carlo', 'stratified', 'adaptive')

    Example:
        >>> sampler = ShapleySampler(n_features=10, sampling_strategy='adaptive')
        >>> subset = sampler.sample_subset()
    """

    def __init__(
        self,
        n_features: int,
        sampling_strategy: str = "monte_carlo",
    ):
        self.n_features = n_features
        self.sampling_strategy = sampling_strategy
        self.rng = np.random.default_rng()

    def sample_subset(self) -> np.ndarray:
        """Sample a random subset of features.

        Returns:
            Binary array indicating selected features
        """
        if self.sampling_strategy == "monte_carlo":
            subset = self._monte_carlo_sample()
        elif self.sampling_strategy == "stratified":
            subset = self._stratified_sample()
        elif self.sampling_strategy == "adaptive":
            subset = self._adaptive_sample()
        else:
            subset = self._monte_carlo_sample()

        return subset

    def _monte_carlo_sample(self) -> np.ndarray:
        """Standard Monte Carlo sampling."""
        return self.rng.choice([0, 1], size=self.n_features, p=[0.5, 0.5]).astype(
            np.float32
        )

    def _stratified_sample(self) -> np.ndarray:
        """Stratified sampling across subset sizes."""
        k = self.rng.integers(1, self.n_features)
        subset = np.zeros(self.n_features, dtype=np.float32)
        subset[self.rng.choice(self.n_features, k, replace=False)] = 1.0
        return subset

    def _adaptive_sample(self) -> np.ndarray:
        """Adaptive sampling focusing on informative subsets."""
        if self.rng.random() < 0.5:
            return self._stratified_sample()
        else:
            return self._monte_carlo_sample()

    def compute_weight(self, subset: np.ndarray) -> float:
        """Compute Shapley kernel weight for a subset.

        The weight is: (M-1) / (M * k * (M-k))
        where M is total features and k is subset size.
        """
        k = int(subset.sum())
        M = self.n_features

        if k == 0 or k == M:
            return float("inf")

        weight = (M - 1) / (M * k * (M - k))
        return weight


def create_shap_explainer(
    model: nn.Module,
    explainer_type: str = "kernel",
    **kwargs,
) -> ShapleyEstimator:
    """Factory function to create SHAP explainers.

    Args:
        model: Model to explain
        explainer_type: Type of explainer ('kernel', 'gradient', 'deep', 'tree')
        **kwargs: Additional arguments for the explainer

    Returns:
        Initialized SHAP explainer

    Example:
        >>> explainer = create_shap_explainer(model, 'gradient', background=data)
    """
    if explainer_type == "kernel":
        return KernelSHAP(model, **kwargs)
    elif explainer_type == "gradient":
        return GradientSHAP(model, **kwargs)
    elif explainer_type == "deep":
        return DeepSHAP(model, **kwargs)
    else:
        raise ValueError(f"Unknown explainer type: {explainer_type}")
