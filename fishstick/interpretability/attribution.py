"""
Attribution Methods

Feature attribution and explanation methods.
"""

from typing import Optional, Tuple, Callable
import torch
from torch import Tensor, nn
import numpy as np


class SaliencyMap:
    """Compute saliency maps using gradients."""

    def __init__(self, model: nn.Module):
        self.model = model

    def __call__(self, x: Tensor, target_class: Optional[int] = None) -> Tensor:
        x.requires_grad = True

        output = self.model(x)

        if target_class is None:
            target_class = output.argmax(dim=1).item()

        self.model.zero_grad()
        output[0, target_class].backward()

        saliency = x.grad.abs()
        return saliency.squeeze()


class IntegratedGradients:
    """Integrated Gradients attribution method."""

    def __init__(
        self, model: nn.Module, baseline: Optional[Tensor] = None, steps: int = 50
    ):
        self.model = model
        self.baseline = baseline
        self.steps = steps

    def __call__(self, x: Tensor, target_class: Optional[int] = None) -> Tensor:
        if self.baseline is None:
            baseline = torch.zeros_like(x)
        else:
            baseline = self.baseline

        scaled_inputs = [
            baseline + (x - baseline) * i / self.steps for i in range(self.steps + 1)
        ]

        gradients = []
        for inp in scaled_inputs:
            inp_grad = inp.clone().detach().requires_grad_(True)
            self.model.zero_grad()

            output = self.model(inp_grad)

            if target_class is None:
                target_class = output.argmax(dim=1).item()

            if output.dim() > 1 and output.shape[1] > 1:
                score = output[0, target_class]
            else:
                score = output[0]

            score.backward()

            if inp_grad.grad is not None:
                gradients.append(inp_grad.grad.clone())

        if not gradients:
            return torch.zeros_like(x)

        avg_gradients = torch.stack(gradients).mean(dim=0)
        attribution = (x - baseline) * avg_gradients

        return attribution.abs()


class SHAPValues:
    """SHAP values approximation."""

    def __init__(self, model: nn.Module, background_size: int = 10):
        self.model = model
        self.background_size = background_size

    def __call__(
        self, x: Tensor, target_class: Optional[int] = None, n_samples: int = 100
    ) -> Tensor:
        x.requires_grad = True

        baseline = torch.zeros_like(x)
        x.requires_grad = True

        output = self.model(x)

        if target_class is None:
            target_class = output.argmax(dim=1).item()

        self.model.zero_grad()
        output[0, target_class].backward()

        shap_values = x.grad * (x - baseline)

        return shap_values.abs()


class LIMEExplainer:
    """LIME (Local Interpretable Model-agnostic Explanations) explainer."""

    def __init__(self, model: nn.Module, perturb_ratio: float = 0.1):
        self.model = model
        self.perturb_ratio = perturb_ratio

    def explain(
        self,
        x: Tensor,
        target_class: Optional[int] = None,
        n_samples: int = 100,
    ) -> Tuple[Tensor, np.ndarray]:
        original_shape = x.shape
        x_flat = x.flatten().unsqueeze(0)

        perturbations = []
        predictions = []

        for _ in range(n_samples):
            mask = (torch.rand_like(x_flat) > self.perturb_ratio).float()
            perturbed = x_flat * mask

            with torch.no_grad():
                output = self.model(perturbed.reshape(original_shape))

            if target_class is None:
                target_class = output.argmax(dim=1).item()

            prob = torch.softmax(output, dim=1)[0, target_class].item()
            perturbations.append(mask.squeeze().numpy())
            predictions.append(prob)

        import scipy.sparse
        from sklearn.linear_model import Lasso

        X = np.array(perturbations)
        y = np.array(predictions)

        model = Lasso(alpha=0.1)
        model.fit(X, y)

        importance = np.abs(model.coef_)
        importance = importance / importance.sum()

        return torch.from_numpy(importance.reshape(original_shape)), importance


class GradCAM:
    """Gradient-weighted Class Activation Mapping."""

    def __init__(self, model: nn.Module, target_layer: Optional[str] = None):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None

    def __call__(self, x: Tensor, target_class: Optional[int] = None) -> Tensor:
        x.requires_grad = True

        def forward_hook(module, input, output):
            self.activations = output

        def backward_hook(module, grad_input, grad_output):
            self.gradients = grad_output[0]

        for name, module in self.model.named_modules():
            if self.target_layer is None or self.target_layer in name:
                module.register_forward_hook(forward_hook)
                module.register_full_backward_hook(backward_hook)

        output = self.model(x)

        if target_class is None:
            target_class = output.argmax(dim=1).item()

        self.model.zero_grad()
        output[0, target_class].backward()

        pooled_gradients = torch.mean(self.gradients, dim=[0, 2, 3])
        for i in range(self.activations.shape[1]):
            self.activations[:, i, :, :] *= pooled_gradients[i]

        heatmap = torch.mean(self.activations, dim=1).squeeze()
        heatmap = torch.relu(heatmap)
        heatmap = heatmap / heatmap.max()

        return heatmap


class OcclusionSensitivity:
    """Occlusion-based sensitivity analysis."""

    def __init__(self, model: nn.Module, window_size: int = 8, stride: int = 4):
        self.model = model
        self.window_size = window_size
        self.stride = stride

    def __call__(self, x: Tensor, target_class: Optional[int] = None) -> Tensor:
        original_pred = self.model(x).argmax(dim=1).item()

        heatmap = torch.zeros(x.shape[2], x.shape[3])

        for i in range(0, x.shape[2] - self.window_size, self.stride):
            for j in range(0, x.shape[3] - self.window_size, self.stride):
                occluded = x.clone()
                occluded[:, :, i : i + self.window_size, j : j + self.window_size] = 0

                with torch.no_grad():
                    new_pred = self.model(occluded).argmax(dim=1).item()

                heatmap[i : i + self.window_size, j : j + self.window_size] = (
                    original_pred - new_pred
                )

        return heatmap


class SmoothGrad:
    """
    SmoothGrad: Reduce noise in saliency maps by averaging over noisy samples.

    Reference: Smilkov et al., "SmoothGrad: removing noise by adding noise", 2017
    """

    def __init__(
        self, model: nn.Module, n_samples: int = 50, noise_level: float = 0.15
    ):
        self.model = model
        self.n_samples = n_samples
        self.noise_level = noise_level

    def __call__(self, x: Tensor, target_class: Optional[int] = None) -> Tensor:
        """Compute SmoothGrad saliency map."""
        saliency = torch.zeros_like(x)

        stdev = self.noise_level * (x.max() - x.min())

        for _ in range(self.n_samples):
            noise = torch.randn_like(x) * stdev
            noisy_x = (x + noise).detach().requires_grad_(True)

            # Compute vanilla gradient on noisy input
            self.model.zero_grad()

            output = self.model(noisy_x)

            if target_class is None:
                target_class = output.argmax(dim=1).item()

            if output.dim() > 1 and output.shape[1] > 1:
                score = output[0, target_class]
            else:
                score = output[0]

            score.backward()

            if noisy_x.grad is not None:
                saliency += noisy_x.grad.abs()

        saliency = saliency / self.n_samples
        return saliency.squeeze()


class DeepLIFT:
    """
    DeepLIFT: Learning Important Features Through Propagating Activation Differences.

    Reference: Shrikumar et al., "Learning Important Features Through
               Propagating Activation Differences", 2017
    """

    def __init__(self, model: nn.Module, baseline: Optional[Tensor] = None):
        self.model = model
        self.baseline = baseline

    def __call__(self, x: Tensor, target_class: Optional[int] = None) -> Tensor:
        """Compute DeepLIFT attributions."""
        if self.baseline is None:
            baseline = torch.zeros_like(x)
        else:
            baseline = self.baseline

        # Compute difference from baseline using gradient approximation
        x.requires_grad = True
        self.model.zero_grad()

        output = self.model(x)

        if target_class is None:
            target_class = output.argmax(dim=1).item()

        output[0, target_class].backward()

        # Approximate DeepLIFT as gradient * (input - baseline)
        if x.grad is not None:
            attributions = x.grad * (x - baseline)
        else:
            attributions = torch.zeros_like(x)

        return attributions.abs().squeeze()


class LayerwiseRelevancePropagation:
    """
    Layer-wise Relevance Propagation (LRP).

    Propagates relevance backwards through the network layer by layer.

    Reference: Bach et al., "On Pixel-Wise Explanations for Non-Linear
               Classifier Decisions", 2015
    """

    def __init__(self, model: nn.Module, epsilon: float = 1e-9):
        self.model = model
        self.epsilon = epsilon
        self.activations = []
        self.handlers = []

    def _register_hooks(self):
        """Register forward hooks to capture activations."""

        def hook_fn(module, input, output):
            self.activations.append(
                (module, input[0].detach() if input else None, output.detach())
            )

        for module in self.model.modules():
            if isinstance(module, (nn.Linear, nn.Conv2d, nn.ReLU, nn.MaxPool2d)):
                handler = module.register_forward_hook(hook_fn)
                self.handlers.append(handler)

    def _remove_hooks(self):
        """Remove all registered hooks."""
        for handler in self.handlers:
            handler.remove()
        self.handlers = []

    def __call__(self, x: Tensor, target_class: Optional[int] = None) -> Tensor:
        """Compute LRP attributions."""
        self.activations = []
        self._register_hooks()

        # Forward pass
        output = self.model(x)

        self._remove_hooks()

        # Initialize relevance at output
        if target_class is None:
            target_class = output.argmax(dim=1).item()

        # For simplicity, use gradient-based approximation
        # Full LRP requires custom backward pass for each layer type
        x.requires_grad = True
        self.model.zero_grad()

        output = self.model(x)
        output[0, target_class].backward()

        if x.grad is not None:
            # Relevance is proportional to gradient times input
            relevance = (x * x.grad).abs()
        else:
            relevance = torch.zeros_like(x)

        return relevance.squeeze()


class NoiseTunnel:
    """
    Noise Tunnel: General framework for smoothing attribution methods.

    Can wrap any attribution method to apply SmoothGrad-style noise smoothing.
    """

    def __init__(
        self, attribution_fn: Callable, n_samples: int = 10, noise_level: float = 0.1
    ):
        self.attribution_fn = attribution_fn
        self.n_samples = n_samples
        self.noise_level = noise_level

    def __call__(self, x: Tensor, target_class: Optional[int] = None) -> Tensor:
        """Compute smoothed attributions."""
        stdev = self.noise_level * (x.max() - x.min())

        attributions = []
        for _ in range(self.n_samples):
            noise = torch.randn_like(x) * stdev
            noisy_x = x + noise
            attr = self.attribution_fn(noisy_x, target_class)
            attributions.append(attr)

        return torch.stack(attributions).mean(dim=0)
