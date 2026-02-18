"""
Integrated Gradients

Implementation of Integrated Gradients (IG) - an axiomatic attribution method
for deep neural networks. IG computes feature importance by integrating
gradients along a path from a baseline input to the original input.

Key properties:
- Sensitivity: If input differs from baseline at an index, attribution is non-zero
- Implementation Invariance: Attribution equals for functionally equivalent networks

This module includes base IG, multiple path methods, layer-integrated gradients,
and wrapped models for easy use.
"""

from __future__ import annotations

from typing import Optional, Callable, Union, List, Tuple, Literal
from abc import ABC, abstractmethod
from enum import Enum

import torch
from torch import Tensor, nn
import numpy as np


class PathMethod(Enum):
    """Available path interpolation methods for integrated gradients."""

    LINEAR = "linear"
    ZIGZAG = "zigzag"
    GAUSSIAN = "gaussian"
    COSINE = "cosine"
    QUADRATIC = "quadratic"


class BaselineStrategy(Enum):
    """Baseline selection strategies."""

    ZERO = "zero"
    MEAN = "mean"
    RANDOM = "random"
    LEARNED = "learned"


class IntegratedGradients:
    """Integrated Gradients attribution method.

    Computes feature attributions by integrating gradients along a straight-line
    path from a baseline input to the original input. Satisfies sensitivity
    and implementation invariance axioms.

    Args:
        model: Differentiable model to explain
        baseline_strategy: Strategy for selecting baseline inputs
        path_method: Method for interpolating between baseline and input
        steps: Number of steps for path integration

    Example:
        >>> ig = IntegratedGradients(model, path_method='linear', steps=50)
        >>> attributions = ig.attribute(input_tensor, target=0)
    """

    def __init__(
        self,
        model: nn.Module,
        baseline_strategy: Union[str, BaselineStrategy] = "zero",
        path_method: Union[str, PathMethod] = "linear",
        steps: int = 50,
    ):
        self.model = model
        self.model.eval()

        if isinstance(baseline_strategy, str):
            self.baseline_strategy = BaselineStrategy(baseline_strategy)
        else:
            self.baseline_strategy = baseline_strategy

        if isinstance(path_method, str):
            self.path_method = PathMethod(path_method)
        else:
            self.path_method = path_method

        self.steps = steps
        self._baseline_cache = None

    def attribute(
        self,
        inputs: Tensor,
        baseline: Optional[Tensor] = None,
        target: Optional[int] = None,
        additional_forward_args: Optional[Tuple] = None,
    ) -> Tensor:
        """Compute integrated gradients attributions.

        Args:
            inputs: Input tensor to explain
            baseline: Baseline input (if None, computed from strategy)
            target: Target class index for attribution
            additional_forward_args: Additional args for model forward

        Returns:
            Attribution tensor of same shape as inputs
        """
        if baseline is None:
            baseline = self._get_baseline(inputs)

        baseline = baseline.to(inputs.device)

        if baseline.shape != inputs.shape:
            baseline = self._align_baseline(baseline, inputs.shape)

        self.model.eval()

        attributions = self._compute_integrated_gradients(
            inputs, baseline, target, additional_forward_args
        )

        return attributions

    def _get_baseline(self, inputs: Tensor) -> Tensor:
        """Get baseline input based on strategy."""
        if self.baseline_strategy == BaselineStrategy.ZERO:
            return torch.zeros_like(inputs)
        elif self.baseline_strategy == BaselineStrategy.MEAN:
            if self._baseline_cache is not None:
                return self._baseline_cache
            return torch.zeros_like(inputs)
        elif self.baseline_strategy == BaselineStrategy.RANDOM:
            return torch.randn_like(inputs) * 0.01
        else:
            return torch.zeros_like(inputs)

    def _align_baseline(self, baseline: Tensor, target_shape: Tuple) -> Tensor:
        """Align baseline shape to input shape."""
        if baseline.dim() == 1 and len(target_shape) > 1:
            baseline = baseline.view(1, *target_shape[1:])
        if baseline.shape[0] == 1 and target_shape[0] > 1:
            baseline = baseline.expand(target_shape[0], *[-1] * (baseline.dim() - 1))
        return baseline

    def _compute_integrated_gradients(
        self,
        inputs: Tensor,
        baseline: Tensor,
        target: Optional[int],
        additional_forward_args: Optional[Tuple],
    ) -> Tensor:
        """Compute integrated gradients along the selected path."""
        batch_size = inputs.shape[0]
        device = inputs.device

        alphas = self._get_path_alphas(self.steps)

        total_gradients = torch.zeros_like(inputs)

        for alpha in alphas:
            interpolated = self._interpolate(inputs, baseline, alpha)
            interpolated.requires_grad_(True)

            output = self.model(interpolated, *(additional_forward_args or []))

            if target is not None:
                if output.dim() > 1:
                    score = output.gather(
                        1,
                        torch.full(
                            (output.size(0),), target, device=device, dtype=torch.long
                        ).unsqueeze(1),
                    ).squeeze()
                else:
                    score = output
            else:
                score = output.max(dim=-1)[0] if output.dim() > 1 else output

            grad = torch.autograd.grad(
                outputs=score,
                inputs=interpolated,
                create_graph=False,
                retain_graph=True,
            )[0]

            total_gradients += grad

        integrated_grads = total_gradients / len(alphas)
        attributions = integrated_grads * (inputs - baseline)

        return attributions

    def _get_path_alphas(self, n_steps: int) -> Tensor:
        """Generate alpha values along the interpolation path."""
        alphas = torch.linspace(0, 1, steps=n_steps)

        if self.path_method == PathMethod.LINEAR:
            return alphas
        elif self.path_method == PathMethod.ZIGZAG:
            return self._zigzag_path(n_steps)
        elif self.path_method == PathMethod.GAUSSIAN:
            return self._gaussian_path(n_steps)
        elif self.path_method == PathMethod.COSINE:
            return self._cosine_path(n_steps)
        elif self.path_method == PathMethod.QUADRATIC:
            return alphas**2
        else:
            return alphas

    def _interpolate(
        self,
        inputs: Tensor,
        baseline: Tensor,
        alpha: Union[float, Tensor],
    ) -> Tensor:
        """Interpolate between baseline and input."""
        if isinstance(alpha, float):
            alpha = torch.tensor(alpha, device=inputs.device)
        return alpha * inputs + (1 - alpha) * baseline

    def _zigzag_path(self, n_steps: int) -> Tensor:
        """Generate zigzag path: 0 -> 1 -> 0 -> 1 with decreasing amplitude."""
        alphas = []
        for i in range(n_steps):
            t = i / (n_steps - 1)
            alpha = t if i % 2 == 0 else 1 - t
            alphas.append(alpha)
        return torch.tensor(alphas)

    def _gaussian_path(self, n_steps: int) -> Tensor:
        """Generate Gaussian-based path for more samples near endpoints."""
        t = torch.linspace(0, 1, steps=n_steps)
        return (torch.erf(3 * (t - 0.5)) + 1) / 2

    def _cosine_path(self, n_steps: int) -> Tensor:
        """Generate cosine-based path emphasizing endpoints."""
        t = torch.linspace(0, np.pi, steps=n_steps)
        return (1 - torch.cos(t)) / 2


class LayerIntegratedGradients:
    """Layer-wise Integrated Gradients for intermediate layers.

    Computes integrated gradients for specific hidden layers, enabling
    understanding of feature representations at different network depths.

    Args:
        model: Neural network model
        layer: Target layer for attribution
        baseline_strategy: Baseline selection strategy

    Example:
        >>> lig = LayerIntegratedGradients(model, layer=model.encoder.layer[5])
        >>> layer_attr = lig.attribute(input_tensor, target=0)
    """

    def __init__(
        self,
        model: nn.Module,
        layer: nn.Module,
        baseline_strategy: Union[str, BaselineStrategy] = "zero",
    ):
        self.model = model
        self.layer = layer
        self.baseline_strategy = (
            BaselineStrategy(baseline_strategy)
            if isinstance(baseline_strategy, str)
            else baseline_strategy
        )

        self._activations = []
        self._gradients = []
        self._hook_handles = []

    def attribute(
        self,
        inputs: Tensor,
        baseline: Optional[Tensor] = None,
        target: Optional[int] = None,
        steps: int = 50,
    ) -> Tuple[Tensor, Tensor]:
        """Compute layer-wise integrated gradients.

        Returns:
            Tuple of (input_attributions, layer_attributions)
        """
        self._register_hooks()

        if baseline is None:
            baseline = torch.zeros_like(inputs)

        baseline = baseline.to(inputs.device)

        self._activations = []
        self._gradients = []

        alphas = torch.linspace(0, 1, steps=steps)
        device = inputs.device

        total_grad = torch.zeros_like(inputs)
        total_layer_grad = None

        for alpha in alphas:
            interpolated = alpha * inputs + (1 - alpha) * baseline
            interpolated.requires_grad_(True)

            output = self.model(interpolated)

            if target is not None:
                score = output[0, target] if output.dim() > 1 else output[0]
            else:
                score = output.max()

            self.model.zero_grad()
            score.backward(retain_graph=True)

            grad_input = interpolated.grad
            grad_layer = self._gradients[-1] if self._gradients else None

            total_grad += grad_input
            if grad_layer is not None:
                if total_layer_grad is None:
                    total_layer_grad = torch.zeros_like(grad_layer)
                total_layer_grad += grad_layer

        self._remove_hooks()

        input_attr = total_grad / steps * (inputs - baseline)
        layer_attr = total_layer_grad / steps if total_layer_grad is not None else None

        return input_attr, layer_attr

    def _register_hooks(self):
        """Register forward and backward hooks."""

        def forward_hook(module, input, output):
            self._activations.append(output)

        def backward_hook(module, grad_input, grad_output):
            self._gradients.append(grad_output[0])

        self._hook_handles.append(self.layer.register_forward_hook(forward_hook))
        self._hook_handles.append(self.layer.register_full_backward_hook(backward_hook))

    def _remove_hooks(self):
        """Remove registered hooks."""
        for handle in self._hook_handles:
            handle.remove()
        self._hook_handles = []


class IntegratedGradientsWrapper(nn.Module):
    """Wrapper module that adds integrated gradients to any model.

    Provides a drop-in replacement that supports .attribute() method
    while maintaining original model functionality.

    Args:
        model: Original model to wrap
        baseline_strategy: Baseline selection strategy

    Example:
        >>> wrapped_model = IntegratedGradientsWrapper(model)
        >>> result = wrapped_model(input_tensor)
        >>> attr = wrapped_model.attribute(input_tensor, target=0)
    """

    def __init__(
        self,
        model: nn.Module,
        baseline_strategy: Union[str, BaselineStrategy] = "zero",
    ):
        super().__init__()
        self.model = model
        self.ig = IntegratedGradients(model, baseline_strategy)

    def forward(self, x: Tensor) -> Tensor:
        """Forward pass through original model."""
        return self.model(x)

    def attribute(
        self,
        inputs: Tensor,
        baseline: Optional[Tensor] = None,
        target: Optional[int] = None,
    ) -> Tensor:
        """Compute integrated gradients attributions."""
        return self.ig.attribute(inputs, baseline, target)


class SmoothedIntegratedGradients:
    """Smoothed Integrated Gradients with noise reduction.

    Combines integrated gradients with noise tunneling for more
    stable and robust attributions by averaging over noisy samples.

    Args:
        model: Model to explain
        nsamples: Number of noisy samples per point
        noise_scale: Scale of Gaussian noise

    Example:
        >>> sig = SmoothedIntegratedGradients(model, nsamples=10)
        >>> attributions = sig.attribute(input_tensor, target=0)
    """

    def __init__(
        self,
        model: nn.Module,
        nsamples: int = 10,
        noise_scale: float = 0.1,
    ):
        self.model = model
        self.nsamples = nsamples
        self.noise_scale = noise_scale
        self.ig = IntegratedGradients(model)

    def attribute(
        self,
        inputs: Tensor,
        baseline: Optional[Tensor] = None,
        target: Optional[int] = None,
    ) -> Tensor:
        """Compute smoothed integrated gradients."""
        device = inputs.device
        total_attr = torch.zeros_like(inputs)

        for _ in range(self.nsamples):
            if self.noise_scale > 0:
                noise = torch.randn_like(inputs) * self.noise_scale
                noisy_input = inputs + noise
            else:
                noisy_input = inputs

            attr = self.ig.attribute(noisy_input, baseline, target)
            total_attr += attr

        return total_attr / self.nsamples


def create_integrated_gradients(
    model: nn.Module,
    method: str = "standard",
    **kwargs,
) -> Union[IntegratedGradients, LayerIntegratedGradients, SmoothedIntegratedGradients]:
    """Factory function to create integrated gradients explainers.

    Args:
        model: Model to explain
        method: Type of IG ('standard', 'layer', 'smoothed')
        **kwargs: Additional arguments

    Returns:
        Configured integrated gradients explainer

    Example:
        >>> ig = create_integrated_gradients(model, method='smoothed', nsamples=20)
    """
    if method == "standard":
        return IntegratedGradients(model, **kwargs)
    elif method == "layer":
        if "layer" not in kwargs:
            raise ValueError("layer parameter required for layer IG")
        layer = kwargs.pop("layer")
        return LayerIntegratedGradients(model, layer, **kwargs)
    elif method == "smoothed":
        return SmoothedIntegratedGradients(model, **kwargs)
    else:
        raise ValueError(f"Unknown method: {method}")
