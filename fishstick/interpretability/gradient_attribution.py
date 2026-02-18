"""
Gradient-based Attribution Methods

Implements various gradient-based interpretability techniques:
- Vanilla Gradients (Saliency Maps)
- SmoothGrad
- Integrated Gradients
- Guided Backpropagation
"""

from typing import Optional, Callable, List, Union, Tuple
import torch
from torch import nn, Tensor
import torch.nn.functional as F
from abc import ABC, abstractmethod


class GradientAttributionBase(ABC):
    """Base class for gradient-based attribution methods."""

    def __init__(self, model: nn.Module):
        self.model = model
        self.model.eval()
        self._hooks = []
        self._activations = {}
        self._gradients = {}

    def _register_hooks(self, target_layer: Optional[nn.Module] = None):
        self._clear_hooks()

        def forward_hook(module, inp, out):
            self._activations["target"] = out.detach()

        def backward_hook(module, grad_in, grad_out):
            self._gradients["target"] = grad_out[0].detach()

        if target_layer is not None:
            h1 = target_layer.register_forward_hook(forward_hook)
            h2 = target_layer.register_full_backward_hook(backward_hook)
            self._hooks = [h1, h2]

    def _clear_hooks(self):
        for h in self._hooks:
            h.remove()
        self._hooks = []
        self._activations = {}
        self._gradients = {}

    @abstractmethod
    def attribute(self, x: Tensor, target: Optional[int] = None) -> Tensor:
        raise NotImplementedError

    def __del__(self):
        self._clear_hooks()


class VanillaGradients(GradientAttributionBase):
    """Vanilla gradients (saliency maps) for attribution.

    Computes gradients of the output with respect to input.

    Args:
        model: PyTorch model
    """

    def __init__(self, model: nn.Module):
        super().__init__(model)

    def attribute(
        self,
        x: Tensor,
        target: Optional[Union[int, Tensor]] = None,
        abs_grad: bool = True,
    ) -> Tensor:
        x = x.clone().requires_grad_(True)

        output = self.model(x)

        if target is None:
            target_indices = output.argmax(dim=-1)
        elif isinstance(target, int):
            target_indices = torch.full(
                (x.size(0),), target, dtype=torch.long, device=x.device
            )
        else:
            target_indices = target

        target_output = output.gather(1, target_indices.unsqueeze(1)).squeeze(1)

        self.model.zero_grad()
        target_output.sum().backward(retain_graph=True)

        gradients = x.grad.clone()

        if abs_grad:
            gradients = gradients.abs()

        return gradients


class SmoothGrad(GradientAttributionBase):
    """SmoothGrad for noise-robust attribution.

    Averages gradients over noisy samples to reduce noise.

    Args:
        model: PyTorch model
        n_samples: Number of noisy samples
        noise_level: Standard deviation of noise
    """

    def __init__(self, model: nn.Module, n_samples: int = 50, noise_level: float = 0.1):
        super().__init__(model)
        self.n_samples = n_samples
        self.noise_level = noise_level

    def attribute(
        self,
        x: Tensor,
        target: Optional[Union[int, Tensor]] = None,
        abs_grad: bool = True,
    ) -> Tensor:
        x = x.clone()
        gradient_sum = torch.zeros_like(x)

        for _ in range(self.n_samples):
            noise = torch.randn_like(x) * self.noise_level
            noisy_input = (x + noise).clone().requires_grad_(True)

            output = self.model(noisy_input)

            if target is None:
                target_indices = output.argmax(dim=-1)
            elif isinstance(target, int):
                target_indices = torch.full(
                    (x.size(0),), target, dtype=torch.long, device=x.device
                )
            else:
                target_indices = target

            target_output = output.gather(1, target_indices.unsqueeze(1)).squeeze(1)

            self.model.zero_grad()
            target_output.sum().backward(retain_graph=True)

            gradient_sum += noisy_input.grad.clone()

        gradients = gradient_sum / self.n_samples

        if abs_grad:
            gradients = gradients.abs()

        return gradients


class IntegratedGradients(GradientAttributionBase):
    """Integrated Gradients for attribution.

    Computes path integral of gradients from baseline to input.

    Args:
        model: PyTorch model
        n_steps: Number of integration steps
        baseline: Baseline input (default: zeros)
    """

    def __init__(
        self, model: nn.Module, n_steps: int = 50, baseline: Optional[Tensor] = None
    ):
        super().__init__(model)
        self.n_steps = n_steps
        self.baseline = baseline

    def attribute(
        self,
        x: Tensor,
        target: Optional[Union[int, Tensor]] = None,
        baseline: Optional[Tensor] = None,
        return_convergence_delta: bool = False,
    ) -> Union[Tensor, Tuple[Tensor, Tensor]]:
        x = x.clone()

        if baseline is None:
            baseline = self.baseline
        if baseline is None:
            baseline = torch.zeros_like(x)

        scaled_inputs = [
            baseline + (float(i) / self.n_steps) * (x - baseline)
            for i in range(self.n_steps + 1)
        ]
        scaled_inputs = torch.stack(scaled_inputs, dim=0)
        scaled_inputs = scaled_inputs.requires_grad_(True)

        original_shape = x.shape
        batch_size = original_shape[0]

        scaled_inputs_flat = scaled_inputs.view(-1, *original_shape[1:])

        output = self.model(scaled_inputs_flat)

        if target is None:
            with torch.no_grad():
                orig_output = self.model(x)
                target_indices = orig_output.argmax(dim=-1)
        elif isinstance(target, int):
            target_indices = torch.full(
                (batch_size,), target, dtype=torch.long, device=x.device
            )
        else:
            target_indices = target

        target_indices_expanded = target_indices.repeat(self.n_steps + 1)
        target_output = output.gather(1, target_indices_expanded.unsqueeze(1)).squeeze(
            1
        )

        self.model.zero_grad()
        target_output.sum().backward()

        gradients = scaled_inputs.grad.clone()
        gradients = gradients.view(self.n_steps + 1, *original_shape)

        avg_gradients = gradients.mean(dim=0)

        attributions = (x - baseline) * avg_gradients

        if return_convergence_delta:
            with torch.no_grad():
                delta = self._compute_convergence_delta(
                    attributions, baseline, x, target_indices
                )
            return attributions, delta

        return attributions

    def _compute_convergence_delta(
        self, attributions: Tensor, baseline: Tensor, input: Tensor, target: Tensor
    ) -> Tensor:
        with torch.no_grad():
            baseline_output = self.model(baseline)
            input_output = self.model(input)

            baseline_target = baseline_output.gather(1, target.unsqueeze(1)).squeeze(1)
            input_target = input_output.gather(1, target.unsqueeze(1)).squeeze(1)

            expected = input_target - baseline_target
            actual = attributions.sum(dim=tuple(range(1, attributions.dim())))

            delta = (actual - expected).abs()

        return delta


class GuidedBackprop(GradientAttributionBase):
    """Guided Backpropagation for visualization.

    Only backpropagates positive gradients through ReLU layers.

    Args:
        model: PyTorch model
    """

    def __init__(self, model: nn.Module):
        super().__init__(model)
        self._relu_hooks = []
        self._modify_relu_gradients()

    def _modify_relu_gradients(self):
        def relu_backward_hook(module, grad_in, grad_out):
            if isinstance(module, nn.ReLU):
                return (F.relu(grad_in[0]),)
            return grad_in

        for module in self.model.modules():
            if isinstance(module, nn.ReLU):
                h = module.register_backward_hook(
                    lambda m, gi, go: (F.relu(gi[0]),) if isinstance(m, nn.ReLU) else gi
                )
                self._relu_hooks.append(h)

    def _clear_hooks(self):
        super()._clear_hooks()
        for h in self._relu_hooks:
            h.remove()
        self._relu_hooks = []

    def attribute(
        self,
        x: Tensor,
        target: Optional[Union[int, Tensor]] = None,
        abs_grad: bool = False,
    ) -> Tensor:
        x = x.clone().requires_grad_(True)

        output = self.model(x)

        if target is None:
            target_indices = output.argmax(dim=-1)
        elif isinstance(target, int):
            target_indices = torch.full(
                (x.size(0),), target, dtype=torch.long, device=x.device
            )
        else:
            target_indices = target

        target_output = output.gather(1, target_indices.unsqueeze(1)).squeeze(1)

        self.model.zero_grad()
        target_output.sum().backward()

        gradients = x.grad.clone()

        if abs_grad:
            gradients = gradients.abs()

        return gradients


class GradCAM(GradientAttributionBase):
    """Gradient-weighted Class Activation Mapping.

    Uses gradients flowing into final conv layer for localization.

    Args:
        model: PyTorch model
        target_layer: Target convolutional layer
    """

    def __init__(self, model: nn.Module, target_layer: Optional[nn.Module] = None):
        super().__init__(model)
        self.target_layer = target_layer
        if target_layer is not None:
            self._register_hooks(target_layer)

    def set_target_layer(self, layer: nn.Module):
        self._register_hooks(layer)

    def attribute(
        self,
        x: Tensor,
        target: Optional[Union[int, Tensor]] = None,
        target_layer: Optional[nn.Module] = None,
    ) -> Tensor:
        if target_layer is not None:
            self.set_target_layer(target_layer)

        if self.target_layer is None:
            raise ValueError("Target layer must be specified")

        x = x.clone().requires_grad_(True)

        output = self.model(x)

        if target is None:
            target_indices = output.argmax(dim=-1)
        elif isinstance(target, int):
            target_indices = torch.full(
                (x.size(0),), target, dtype=torch.long, device=x.device
            )
        else:
            target_indices = target

        target_output = output.gather(1, target_indices.unsqueeze(1)).squeeze(1)

        self.model.zero_grad()
        target_output.sum().backward(retain_graph=True)

        activations = self._activations.get("target")
        gradients = self._gradients.get("target")

        if activations is None or gradients is None:
            raise RuntimeError("Could not capture activations/gradients")

        weights = gradients.mean(dim=(2, 3), keepdim=True)
        cam = (weights * activations).sum(dim=1, keepdim=True)
        cam = F.relu(cam)

        cam = F.interpolate(cam, size=x.shape[2:], mode="bilinear", align_corners=False)

        cam_min = cam.amin(dim=(2, 3), keepdim=True)
        cam_max = cam.amax(dim=(2, 3), keepdim=True)
        cam = (cam - cam_min) / (cam_max - cam_min + 1e-8)

        return cam.squeeze(1)


class GradientxSign(GradientAttributionBase):
    """Gradient times Sign attribution method.

    Multiplies input by sign of gradient for attribution.

    Args:
        model: PyTorch model
    """

    def __init__(self, model: nn.Module):
        super().__init__(model)

    def attribute(
        self, x: Tensor, target: Optional[Union[int, Tensor]] = None
    ) -> Tensor:
        x = x.clone().requires_grad_(True)

        output = self.model(x)

        if target is None:
            target_indices = output.argmax(dim=-1)
        elif isinstance(target, int):
            target_indices = torch.full(
                (x.size(0),), target, dtype=torch.long, device=x.device
            )
        else:
            target_indices = target

        target_output = output.gather(1, target_indices.unsqueeze(1)).squeeze(1)

        self.model.zero_grad()
        target_output.sum().backward()

        gradients = x.grad.clone()
        attribution = x.detach() * gradients.sign()

        return attribution


class DeepLIFT(GradientAttributionBase):
    """DeepLIFT attribution method.

    Compares activations to reference activation.

    Args:
        model: PyTorch model
        baseline: Reference input
    """

    def __init__(self, model: nn.Module, baseline: Optional[Tensor] = None):
        super().__init__(model)
        self.baseline = baseline

    def attribute(
        self,
        x: Tensor,
        target: Optional[Union[int, Tensor]] = None,
        baseline: Optional[Tensor] = None,
    ) -> Tensor:
        x = x.clone()

        if baseline is None:
            baseline = self.baseline
        if baseline is None:
            baseline = torch.zeros_like(x)

        x_input = x.clone().requires_grad_(True)
        baseline_input = baseline.clone().requires_grad_(True)

        output = self.model(x_input)
        baseline_output = self.model(baseline_input)

        if target is None:
            target_indices = output.argmax(dim=-1)
        elif isinstance(target, int):
            target_indices = torch.full(
                (x.size(0),), target, dtype=torch.long, device=x.device
            )
        else:
            target_indices = target

        target_output = output.gather(1, target_indices.unsqueeze(1)).squeeze(1)
        baseline_target = baseline_output.gather(
            1, target_indices.unsqueeze(1)
        ).squeeze(1)

        delta = target_output - baseline_target

        self.model.zero_grad()
        delta.sum().backward()

        grad_input = x_input.grad.clone()

        attribution = (x - baseline) * grad_input

        return attribution


def create_gradient_attribution(
    method: str, model: nn.Module, **kwargs
) -> GradientAttributionBase:
    """Factory function to create gradient attribution methods.

    Args:
        method: Method name ('vanilla', 'smoothgrad', 'integrated',
                'guided_backprop', 'gradcam', 'gradxsign', 'deeplift')
        model: PyTorch model
        **kwargs: Additional arguments for specific methods

    Returns:
        Attribution method instance
    """
    methods = {
        "vanilla": VanillaGradients,
        "saliency": VanillaGradients,
        "smoothgrad": SmoothGrad,
        "integrated": IntegratedGradients,
        "guided_backprop": GuidedBackprop,
        "gradcam": GradCAM,
        "gradxsign": GradientxSign,
        "deeplift": DeepLIFT,
    }

    if method.lower() not in methods:
        raise ValueError(f"Unknown method: {method}. Available: {list(methods.keys())}")

    return methods[method.lower()](model, **kwargs)
