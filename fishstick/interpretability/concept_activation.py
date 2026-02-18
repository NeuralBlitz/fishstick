"""
Concept Activation Vectors

Implements concept-based interpretability methods:
- TCAV (Testing with Concept Activation Vectors)
- Concept Discovery
- Concept Bottleneck Models
"""

from typing import Optional, List, Dict, Union, Tuple, Callable
import torch
from torch import nn, Tensor
import torch.nn.functional as F
from abc import ABC, abstractmethod
import math


class ConceptActivationBase(ABC):
    """Base class for concept activation methods."""

    def __init__(self, model: nn.Module, layer: Optional[nn.Module] = None):
        self.model = model
        self.model.eval()
        self.layer = layer
        self._hooks = []
        self._activations = {}

        if layer is not None:
            self._register_hook(layer)

    def _register_hook(self, layer: nn.Module):
        self._clear_hooks()

        def hook(module, inp, out):
            self._activations["target"] = out.detach()

        h = layer.register_forward_hook(hook)
        self._hooks = [h]

    def _clear_hooks(self):
        for h in self._hooks:
            h.remove()
        self._hooks = []
        self._activations = {}

    def set_layer(self, layer: nn.Module):
        self.layer = layer
        self._register_hook(layer)

    def _get_activations(self, x: Tensor) -> Tensor:
        if self.layer is None:
            raise ValueError("Target layer must be set")

        self._activations = {}
        with torch.no_grad():
            _ = self.model(x)

        activations = self._activations.get("target")
        if activations is None:
            raise RuntimeError("Could not capture activations")

        return activations

    def __del__(self):
        self._clear_hooks()


class TCAV(ConceptActivationBase):
    """Testing with Concept Activation Vectors.

    Tests whether a concept is important for a prediction class.

    Args:
        model: PyTorch model
        layer: Target layer for concept analysis
        cav_method: Method to compute CAV ('svm', 'linear', 'logistic')
    """

    def __init__(
        self,
        model: nn.Module,
        layer: Optional[nn.Module] = None,
        cav_method: str = "linear",
    ):
        super().__init__(model, layer)
        self.cav_method = cav_method
        self._cavs: Dict[str, Tensor] = {}
        self._concept_names: List[str] = []

    def learn_concept(
        self,
        concept_examples: Tensor,
        random_examples: Tensor,
        concept_name: Optional[str] = None,
    ) -> Tensor:
        if concept_name is None:
            concept_name = f"concept_{len(self._concept_names)}"

        concept_activations = self._get_activations(concept_examples)
        random_activations = self._get_activations(random_examples)

        if concept_activations.dim() > 2:
            concept_activations = concept_activations.flatten(1)
            random_activations = random_activations.flatten(1)

        X = torch.cat([concept_activations, random_activations], dim=0)
        y = torch.cat(
            [
                torch.ones(concept_activations.size(0)),
                torch.zeros(random_activations.size(0)),
            ],
            dim=0,
        ).to(X.device)

        cav = self._compute_cav(X, y)

        self._cavs[concept_name] = cav
        self._concept_names.append(concept_name)

        return cav

    def _compute_cav(self, X: Tensor, y: Tensor) -> Tensor:
        if self.cav_method == "linear":
            X_mean = X.mean(dim=0)
            X_centered = X - X_mean

            y_centered = (y - 0.5) * 2

            cav = (X_centered.T @ y_centered) / X.size(0)
            cav = cav / (cav.norm() + 1e-8)

        elif self.cav_method == "svm":
            cav = self._svm_cav(X, y)

        elif self.cav_method == "logistic":
            cav = self._logistic_cav(X, y)

        else:
            raise ValueError(f"Unknown CAV method: {self.cav_method}")

        return cav

    def _svm_cav(
        self, X: Tensor, y: Tensor, n_iterations: int = 100, lr: float = 0.01
    ) -> Tensor:
        n_features = X.size(1)
        weights = torch.randn(n_features, device=X.device) * 0.01
        weights.requires_grad_(True)

        optimizer = torch.optim.SGD([weights], lr=lr)

        for _ in range(n_iterations):
            optimizer.zero_grad()

            margins = X @ weights
            hinge_loss = F.relu(1 - y * margins).mean()
            reg_loss = 0.01 * (weights**2).sum()
            loss = hinge_loss + reg_loss

            loss.backward()
            optimizer.step()

        cav = weights.detach()
        cav = cav / (cav.norm() + 1e-8)

        return cav

    def _logistic_cav(
        self, X: Tensor, y: Tensor, n_iterations: int = 100, lr: float = 0.1
    ) -> Tensor:
        n_features = X.size(1)
        weights = torch.randn(n_features, device=X.device) * 0.01
        weights.requires_grad_(True)

        optimizer = torch.optim.SGD([weights], lr=lr)

        for _ in range(n_iterations):
            optimizer.zero_grad()

            logits = X @ weights
            loss = F.binary_cross_entropy_with_logits(logits, y)
            loss += 0.01 * (weights**2).sum()

            loss.backward()
            optimizer.step()

        cav = weights.detach()
        cav = cav / (cav.norm() + 1e-8)

        return cav

    def test_concept(
        self, x: Tensor, concept_name: str, target_class: Optional[int] = None
    ) -> Dict[str, Tensor]:
        if concept_name not in self._cavs:
            raise ValueError(f"Concept '{concept_name}' not learned")

        cav = self._cavs[concept_name]

        x = x.clone().requires_grad_(True)

        output = self.model(x)

        if target_class is None:
            target_class = output.argmax(dim=-1).item()

        target_output = output[:, target_class].sum()

        self.model.zero_grad()
        target_output.backward(retain_graph=True)

        input_grad = x.grad.clone()

        if input_grad.dim() > 2:
            input_grad = input_grad.flatten(1)

        cav_expanded = cav.view(1, -1).expand(input_grad.size(0), -1)

        sensitivity = (input_grad * cav_expanded).sum(dim=1)

        n_samples = x.size(0)
        tcav_score = (sensitivity > 0).float().mean()

        return {
            "sensitivity": sensitivity,
            "tcav_score": tcav_score,
            "concept": concept_name,
            "target_class": target_class,
        }

    def get_concept_vector(self, concept_name: str) -> Tensor:
        if concept_name not in self._cavs:
            raise ValueError(f"Concept '{concept_name}' not learned")
        return self._cavs[concept_name]

    def list_concepts(self) -> List[str]:
        return self._concept_names.copy()


class ConceptDiscovery(ConceptActivationBase):
    """Automatic concept discovery from activations.

    Discovers interpretable concepts from model activations.

    Args:
        model: PyTorch model
        layer: Target layer
        n_concepts: Number of concepts to discover
    """

    def __init__(
        self, model: nn.Module, layer: Optional[nn.Module] = None, n_concepts: int = 10
    ):
        super().__init__(model, layer)
        self.n_concepts = n_concepts
        self._concepts: Dict[int, Tensor] = {}
        self._concept_labels: Dict[int, List[int]] = {}

    def discover_concepts(self, x: Tensor, method: str = "kmeans") -> Dict[int, Tensor]:
        activations = self._get_activations(x)

        if activations.dim() > 2:
            activations = activations.flatten(1)

        if method == "kmeans":
            concepts = self._kmeans_discovery(activations)
        elif method == "pca":
            concepts = self._pca_discovery(activations)
        elif method == "nmf":
            concepts = self._nmf_discovery(activations)
        else:
            raise ValueError(f"Unknown discovery method: {method}")

        self._concepts = concepts
        return concepts

    def _kmeans_discovery(
        self, activations: Tensor, n_iterations: int = 20
    ) -> Dict[int, Tensor]:
        n_samples = activations.size(0)
        n_features = activations.size(1)

        indices = torch.randperm(n_samples)[: self.n_concepts]
        centroids = activations[indices].clone()

        for _ in range(n_iterations):
            distances = torch.cdist(activations, centroids)
            labels = distances.argmin(dim=1)

            new_centroids = torch.zeros_like(centroids)
            for k in range(self.n_concepts):
                mask = labels == k
                if mask.sum() > 0:
                    new_centroids[k] = activations[mask].mean(dim=0)
                else:
                    new_centroids[k] = centroids[k]

            if torch.allclose(centroids, new_centroids, atol=1e-6):
                break
            centroids = new_centroids

        self._concept_labels = {}
        for k in range(self.n_concepts):
            self._concept_labels[k] = torch.where(labels == k)[0].tolist()

        concepts = {}
        for k in range(self.n_concepts):
            concepts[k] = centroids[k] / (centroids[k].norm() + 1e-8)

        return concepts

    def _pca_discovery(self, activations: Tensor) -> Dict[int, Tensor]:
        X_centered = activations - activations.mean(dim=0, keepdim=True)

        cov = X_centered.T @ X_centered / (X_centered.size(0) - 1)

        eigenvalues, eigenvectors = torch.linalg.eigh(cov)

        sorted_indices = eigenvalues.argsort(descending=True)
        top_components = eigenvectors[:, sorted_indices[: self.n_concepts]]

        concepts = {}
        for k in range(self.n_concepts):
            concepts[k] = top_components[:, k]

        return concepts

    def _nmf_discovery(
        self, activations: Tensor, n_iterations: int = 50
    ) -> Dict[int, Tensor]:
        activations = F.relu(activations)

        n_samples = activations.size(0)
        n_features = activations.size(1)

        W = torch.rand(n_samples, self.n_concepts, device=activations.device)
        H = torch.rand(self.n_concepts, n_features, device=activations.device)

        eps = 1e-8

        for _ in range(n_iterations):
            H = H * (W.T @ activations) / (W.T @ W @ H + eps)
            W = W * (activations @ H.T) / (W @ H @ H.T + eps)

        concepts = {}
        for k in range(self.n_concepts):
            concepts[k] = H[k] / (H[k].norm() + 1e-8)

        return concepts

    def get_examples_for_concept(
        self, concept_id: int, x: Tensor, top_k: int = 10
    ) -> Tensor:
        if concept_id not in self._concepts:
            raise ValueError(f"Concept {concept_id} not discovered")

        activations = self._get_activations(x)

        if activations.dim() > 2:
            activations = activations.flatten(1)

        concept_vector = self._concepts[concept_id]
        similarities = F.cosine_similarity(activations, concept_vector.unsqueeze(0))

        top_indices = similarities.topk(min(top_k, x.size(0))).indices

        return x[top_indices]

    def explain_concept(
        self,
        concept_id: int,
        feature_names: Optional[List[str]] = None,
        top_k: int = 10,
    ) -> Dict[str, Union[Tensor, List]]:
        if concept_id not in self._concepts:
            raise ValueError(f"Concept {concept_id} not discovered")

        concept_vector = self._concepts[concept_id]

        top_indices = concept_vector.abs().topk(top_k).indices
        top_values = concept_vector[top_indices]

        explanation = {
            "concept_id": concept_id,
            "top_features": top_indices.tolist(),
            "top_values": top_values.tolist(),
        }

        if feature_names is not None:
            explanation["feature_names"] = [
                feature_names[i] for i in top_indices.tolist()
            ]

        return explanation


class ConceptBottleneckModel(nn.Module):
    """Concept Bottleneck Model.

    Forces predictions through interpretable concept layer.

    Args:
        encoder: Feature encoder
        n_concepts: Number of bottleneck concepts
        n_classes: Number of output classes
        concept_activation: Activation for concepts ('sigmoid', 'relu')
    """

    def __init__(
        self,
        encoder: nn.Module,
        n_concepts: int,
        n_classes: int,
        concept_activation: str = "sigmoid",
    ):
        super().__init__()
        self.encoder = encoder

        with torch.no_grad():
            dummy = torch.randn(1, 3, 224, 224)
            encoder_out = encoder(dummy)
            if isinstance(encoder_out, tuple):
                encoder_dim = encoder_out[0].size(1)
            else:
                encoder_dim = encoder_out.size(1)

        self.concept_layer = nn.Linear(encoder_dim, n_concepts)
        self.classifier = nn.Linear(n_concepts, n_classes)

        self.concept_activation = concept_activation
        self.n_concepts = n_concepts
        self.n_classes = n_classes

    def forward(
        self, x: Tensor, return_concepts: bool = False
    ) -> Union[Tensor, Tuple[Tensor, Tensor]]:
        features = self.encoder(x)

        if isinstance(features, tuple):
            features = features[0]

        features = features.flatten(1)

        concept_logits = self.concept_layer(features)

        if self.concept_activation == "sigmoid":
            concepts = torch.sigmoid(concept_logits)
        elif self.concept_activation == "relu":
            concepts = F.relu(concept_logits)
        else:
            concepts = concept_logits

        logits = self.classifier(concepts)

        if return_concepts:
            return logits, concepts
        return logits

    def intervene(self, x: Tensor, concept_values: Dict[int, float]) -> Tensor:
        _, concepts = self.forward(x, return_concepts=True)

        for idx, value in concept_values.items():
            concepts[:, idx] = value

        logits = self.classifier(concepts)
        return logits

    def get_concept_importance(self) -> Tensor:
        return self.classifier.weight.abs()


class ConceptWhitening(nn.Module):
    """Concept Whitening Layer.

    Whitens activations aligned with concept directions.

    Args:
        num_features: Number of input features
        n_concepts: Number of concepts to whiten
    """

    def __init__(self, num_features: int, n_concepts: int):
        super().__init__()
        self.num_features = num_features
        self.n_concepts = n_concepts

        self.concept_vectors = nn.Parameter(
            torch.randn(n_concepts, num_features) * 0.01, requires_grad=True
        )

        self.running_mean = nn.Parameter(torch.zeros(num_features), requires_grad=False)
        self.running_var = nn.Parameter(torch.ones(num_features), requires_grad=False)

    def forward(self, x: Tensor) -> Tensor:
        if self.training:
            with torch.no_grad():
                self.running_mean.data = 0.9 * self.running_mean.data + 0.1 * x.mean(
                    dim=0
                )
                self.running_var.data = 0.9 * self.running_var.data + 0.1 * x.var(dim=0)

        x_norm = (x - self.running_mean) / (self.running_var + 1e-5).sqrt()

        concept_proj = x_norm @ self.concept_vectors.T

        concept_scores = torch.tanh(concept_proj)

        whitened = x_norm - concept_scores @ self.concept_vectors

        return whitened

    def get_concept_scores(self, x: Tensor) -> Tensor:
        x_norm = (x - self.running_mean) / (self.running_var + 1e-5).sqrt()
        return torch.tanh(x_norm @ self.concept_vectors.T)


class ConceptAlignmentScore:
    """Computes alignment between learned concepts and ground truth.

    Args:
        model: Model with concept layer
    """

    def __init__(self, model: nn.Module):
        self.model = model

    def compute_alignment(
        self,
        x: Tensor,
        ground_truth_concepts: Tensor,
        concept_layer: Optional[nn.Module] = None,
    ) -> Dict[str, Tensor]:
        if concept_layer is not None:
            with torch.no_grad():
                features = self.model.encoder(x)
                if isinstance(features, tuple):
                    features = features[0]
                features = features.flatten(1)
                predicted = torch.sigmoid(concept_layer(features))
        else:
            with torch.no_grad():
                _, predicted = self.model(x, return_concepts=True)

        mse = F.mse_loss(predicted, ground_truth_concepts, reduction="none").mean(dim=0)

        correlations = []
        for i in range(predicted.size(1)):
            pred_i = predicted[:, i]
            gt_i = ground_truth_concepts[:, i]

            pred_centered = pred_i - pred_i.mean()
            gt_centered = gt_i - gt_i.mean()

            corr = (pred_centered * gt_centered).sum() / (
                pred_centered.norm() * gt_centered.norm() + 1e-8
            )
            correlations.append(corr)

        correlations = torch.stack(correlations)

        accuracy = (
            ((predicted > 0.5) == ground_truth_concepts.bool()).float().mean(dim=0)
        )

        return {
            "mse_per_concept": mse,
            "correlation_per_concept": correlations,
            "accuracy_per_concept": accuracy,
            "overall_mse": mse.mean(),
            "overall_correlation": correlations.mean(),
            "overall_accuracy": accuracy.mean(),
        }


class ConceptCompleteness:
    """Evaluates concept completeness for downstream tasks.

    Args:
        model: Concept-based model
    """

    def __init__(self, model: nn.Module):
        self.model = model

    def evaluate(
        self, x: Tensor, concepts: Tensor, targets: Tensor
    ) -> Dict[str, float]:
        with torch.no_grad():
            logits = self.model.classifier(concepts)
            predictions = logits.argmax(dim=-1)

            accuracy = (predictions == targets).float().mean().item()

            proba = F.softmax(logits, dim=-1)
            target_proba = proba.gather(1, targets.unsqueeze(1)).squeeze(1)
            confidence = target_proba.mean().item()

        n_concepts = concepts.size(1)
        n_samples = concepts.size(0)
        concept_usage = (concepts > 0.5).float().mean().item()

        concept_entropy = (
            -(
                concepts * torch.log(concepts + 1e-10)
                + (1 - concepts) * torch.log(1 - concepts + 1e-10)
            )
            .mean()
            .item()
        )

        return {
            "accuracy": accuracy,
            "confidence": confidence,
            "concept_usage": concept_usage,
            "concept_entropy": concept_entropy,
            "n_concepts": n_concepts,
        }


def create_concept_method(
    method: str, model: nn.Module, layer: Optional[nn.Module] = None, **kwargs
) -> ConceptActivationBase:
    """Factory function to create concept activation methods.

    Args:
        method: Method name ('tcav', 'discovery')
        model: PyTorch model
        layer: Target layer
        **kwargs: Additional arguments

    Returns:
        Concept method instance
    """
    methods = {
        "tcav": TCAV,
        "discovery": ConceptDiscovery,
    }

    method_lower = method.lower()
    if method_lower not in methods:
        raise ValueError(f"Unknown method: {method}. Available: {list(methods.keys())}")

    return methods[method_lower](model, layer, **kwargs)
