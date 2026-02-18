"""
Class Activation Mapping (CAM) Methods

Implements various CAM-based visualization techniques:
- GradCAM
- GradCAM++
- ScoreCAM
- EigenCAM
- LayerCAM
"""

from typing import Optional, List, Union, Tuple
import torch
from torch import nn, Tensor
import torch.nn.functional as F
from abc import ABC, abstractmethod
import math


class CAMBase(ABC):
    """Base class for Class Activation Mapping methods."""

    def __init__(self, model: nn.Module, target_layer: Optional[nn.Module] = None):
        self.model = model
        self.model.eval()
        self.target_layer = target_layer
        self._hooks = []
        self._activations = {}
        self._gradients = {}

        if target_layer is not None:
            self._register_hooks(target_layer)

    def _register_hooks(self, target_layer: nn.Module):
        self._clear_hooks()

        def forward_hook(module, inp, out):
            self._activations["target"] = out

        def backward_hook(module, grad_in, grad_out):
            self._gradients["target"] = grad_out[0]

        h1 = target_layer.register_forward_hook(forward_hook)
        h2 = target_layer.register_full_backward_hook(backward_hook)
        self._hooks = [h1, h2]

    def _clear_hooks(self):
        for h in self._hooks:
            h.remove()
        self._hooks = []
        self._activations = {}
        self._gradients = {}

    def set_target_layer(self, layer: nn.Module):
        self.target_layer = layer
        self._register_hooks(layer)

    def _normalize_cam(self, cam: Tensor) -> Tensor:
        cam_min = cam.amin(dim=(-2, -1), keepdim=True)
        cam_max = cam.amax(dim=(-2, -1), keepdim=True)
        return (cam - cam_min) / (cam_max - cam_min + 1e-8)

    def _resize_cam(self, cam: Tensor, target_size: Tuple[int, int]) -> Tensor:
        return F.interpolate(
            cam.unsqueeze(1), size=target_size, mode="bilinear", align_corners=False
        ).squeeze(1)

    @abstractmethod
    def attribute(
        self, x: Tensor, target: Optional[Union[int, Tensor]] = None
    ) -> Tensor:
        raise NotImplementedError

    def __del__(self):
        self._clear_hooks()


class GradCAM(CAMBase):
    """Gradient-weighted Class Activation Mapping.

    Uses gradient information flowing into the final convolutional layer
    to assign importance values to each neuron for a particular decision.

    Args:
        model: PyTorch model
        target_layer: Target convolutional layer
    """

    def __init__(self, model: nn.Module, target_layer: Optional[nn.Module] = None):
        super().__init__(model, target_layer)

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

        cam = self._normalize_cam(cam)

        return self._resize_cam(cam.squeeze(1), x.shape[2:])


class GradCAMPlusPlus(CAMBase):
    """GradCAM++ - Improved GradCAM for better localization.

    Uses second-order gradients for weighted combination of feature maps.

    Args:
        model: PyTorch model
        target_layer: Target convolutional layer
    """

    def __init__(self, model: nn.Module, target_layer: Optional[nn.Module] = None):
        super().__init__(model, target_layer)

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
        target_output.sum().backward(retain_graph=True, create_graph=True)

        activations = self._activations.get("target")
        gradients = self._gradients.get("target")

        if activations is None or gradients is None:
            raise RuntimeError("Could not capture activations/gradients")

        alpha_numer = gradients.pow(2)

        grad_sum = gradients.sum(dim=(2, 3), keepdim=True)
        alpha_denom = 2 * alpha_numer + activations * grad_sum + 1e-8
        alpha = alpha_numer / alpha_denom

        weights = (alpha * F.relu(gradients)).sum(dim=(2, 3), keepdim=True)

        cam = (weights * activations).sum(dim=1, keepdim=True)
        cam = F.relu(cam)

        cam = self._normalize_cam(cam)

        return self._resize_cam(cam.squeeze(1), x.shape[2:])


class ScoreCAM(CAMBase):
    """ScoreCAM - Score-weighted Class Activation Mapping.

    Uses forward pass scores instead of gradients for weighting.

    Args:
        model: PyTorch model
        target_layer: Target convolutional layer
    """

    def __init__(
        self,
        model: nn.Module,
        target_layer: Optional[nn.Module] = None,
        batch_size: int = 32,
    ):
        super().__init__(model, target_layer)
        self.batch_size = batch_size

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

        with torch.no_grad():
            output = self.model(x)

            if target is None:
                target_indices = output.argmax(dim=-1)
            elif isinstance(target, int):
                target_indices = torch.full(
                    (x.size(0),), target, dtype=torch.long, device=x.device
                )
            else:
                target_indices = target

        self._activations = {}
        _ = self.model(x)
        activations = self._activations.get("target")

        if activations is None:
            raise RuntimeError("Could not capture activations")

        B, C, H, W = activations.shape
        batch_size = x.size(0)

        scores = []

        for i in range(C):
            channel_activation = activations[:, i : i + 1, :, :]
            channel_activation = F.interpolate(
                channel_activation,
                size=x.shape[2:],
                mode="bilinear",
                align_corners=False,
            )

            normalized = self._normalize_cam(channel_activation)

            masked_input = x * normalized

            with torch.no_grad():
                masked_output = self.model(masked_input)

            score = masked_output.gather(1, target_indices.unsqueeze(1)).squeeze(1)
            scores.append(score)

        scores = torch.stack(scores, dim=1)
        weights = F.softmax(scores, dim=1)

        cam = (weights.unsqueeze(-1).unsqueeze(-1) * activations).sum(dim=1)
        cam = F.relu(cam)

        cam = self._normalize_cam(cam.unsqueeze(1)).squeeze(1)

        return cam


class EigenCAM(CAMBase):
    """EigenCAM - PCA-based Class Activation Mapping.

    Uses principal components of feature maps for visualization.

    Args:
        model: PyTorch model
        target_layer: Target convolutional layer
    """

    def __init__(self, model: nn.Module, target_layer: Optional[nn.Module] = None):
        super().__init__(model, target_layer)

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

        with torch.no_grad():
            _ = self.model(x)

        activations = self._activations.get("target")

        if activations is None:
            raise RuntimeError("Could not capture activations")

        B, C, H, W = activations.shape

        cams = []
        for b in range(B):
            act = activations[b]
            act_flat = act.view(C, -1)

            act_centered = act_flat - act_flat.mean(dim=1, keepdim=True)
            cov = act_centered @ act_centered.T / (H * W)

            try:
                eigenvalues, eigenvectors = torch.linalg.eigh(cov)

                sorted_idx = eigenvalues.argsort(descending=True)
                first_eigenvector = eigenvectors[:, sorted_idx[0]]

                cam = (first_eigenvector.unsqueeze(1) * act_flat).sum(dim=0)
                cam = cam.view(H, W)
            except RuntimeError:
                cam = act_flat.mean(dim=0).view(H, W)

            cams.append(cam)

        cam = torch.stack(cams)

        cam = F.relu(cam)
        cam = self._normalize_cam(cam.unsqueeze(1)).squeeze(1)

        return self._resize_cam(cam, x.shape[2:])


class LayerCAM(CAMBase):
    """LayerCAM - Layer-wise Class Activation Mapping.

    Considers positive contributions from each location.

    Args:
        model: PyTorch model
        target_layer: Target convolutional layer
    """

    def __init__(self, model: nn.Module, target_layer: Optional[nn.Module] = None):
        super().__init__(model, target_layer)

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

        positive_grads = F.relu(gradients)

        cam = (positive_grads * activations).sum(dim=1, keepdim=True)
        cam = F.relu(cam)

        cam = self._normalize_cam(cam)

        return self._resize_cam(cam.squeeze(1), x.shape[2:])


class XGradCAM(CAMBase):
    """XGradCAM - Axiom-based GradCAM.

    Uses normalized gradients for more stable attribution.

    Args:
        model: PyTorch model
        target_layer: Target convolutional layer
    """

    def __init__(self, model: nn.Module, target_layer: Optional[nn.Module] = None):
        super().__init__(model, target_layer)

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

        sum_activations = activations.sum(dim=(2, 3), keepdim=True)
        normalized_weights = (gradients * activations) / (sum_activations + 1e-8)
        weights = normalized_weights.mean(dim=(2, 3), keepdim=True)

        cam = (weights * activations).sum(dim=1, keepdim=True)
        cam = F.relu(cam)

        cam = self._normalize_cam(cam)

        return self._resize_cam(cam.squeeze(1), x.shape[2:])


class MultiLayerCAM:
    """Multi-layer CAM aggregation.

    Computes CAMs from multiple layers and aggregates them.

    Args:
        model: PyTorch model
        target_layers: List of target layers
        method: CAM method to use ('gradcam', 'gradcam++', 'layercam')
    """

    def __init__(
        self, model: nn.Module, target_layers: List[nn.Module], method: str = "gradcam"
    ):
        self.model = model
        self.target_layers = target_layers
        self.method = method.lower()

        self.cam_methods = {
            "gradcam": GradCAM,
            "gradcam++": GradCAMPlusPlus,
            "gradcampp": GradCAMPlusPlus,
            "layercam": LayerCAM,
            "eigencam": EigenCAM,
            "xgradcam": XGradCAM,
        }

        if self.method not in self.cam_methods:
            raise ValueError(
                f"Unknown method: {method}. Available: {list(self.cam_methods.keys())}"
            )

    def attribute(
        self, x: Tensor, target: Optional[Union[int, Tensor]] = None
    ) -> Tensor:
        cams = []

        for layer in self.target_layers:
            cam_method = self.cam_methods[self.method](self.model, layer)
            cam = cam_method.attribute(x, target)
            cams.append(cam)

        cam = torch.stack(cams).mean(dim=0)
        cam = cam - cam.amin(dim=(-2, -1), keepdim=True)
        cam = cam / (cam.amax(dim=(-2, -1), keepdim=True) + 1e-8)

        return cam


def create_cam(
    method: str, model: nn.Module, target_layer: Optional[nn.Module] = None, **kwargs
) -> CAMBase:
    """Factory function to create CAM methods.

    Args:
        method: Method name ('gradcam', 'gradcam++', 'scorecam',
                'eigencam', 'layercam', 'xgradcam')
        model: PyTorch model
        target_layer: Target convolutional layer
        **kwargs: Additional arguments

    Returns:
        CAM method instance
    """
    methods = {
        "gradcam": GradCAM,
        "gradcam++": GradCAMPlusPlus,
        "gradcampp": GradCAMPlusPlus,
        "scorecam": ScoreCAM,
        "eigencam": EigenCAM,
        "layercam": LayerCAM,
        "xgradcam": XGradCAM,
    }

    method_lower = method.lower()
    if method_lower not in methods:
        raise ValueError(f"Unknown method: {method}. Available: {list(methods.keys())}")

    return methods[method_lower](model, target_layer, **kwargs)
