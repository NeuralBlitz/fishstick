"""
Concept-Based Interpretability

Tools for concept-based explanations.
"""

from typing import Optional, List, Tuple
import torch
from torch import Tensor, nn
import numpy as np
from sklearn.linear_model import Ridge


class ConceptExtractor:
    """Extract human-understandable concepts from model layers."""

    def __init__(self, model: nn.Module, layer_names: List[str]):
        self.model = model
        self.layer_names = layer_names
        self.activations = {}
        self._register_hooks()

    def _register_hooks(self):
        def get_activation(name):
            def hook(module, input, output):
                self.activations[name] = output.detach()

            return hook

        for name, module in self.model.named_modules():
            if any(layer in name for layer in self.layer_names):
                module.register_forward_hook(get_activation(name))

    def extract_concepts(
        self, x: Tensor, n_concepts: int = 10
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Extract concept directions using PCA or ICA."""
        with torch.no_grad():
            self.model(x)

        activations = []
        for name in self.layer_names:
            if name in self.activations:
                act = self.activations[name]
                if act.ndim > 2:
                    act = act.flatten(1)
                activations.append(act)

        if not activations:
            raise ValueError("No activations captured")

        all_activations = torch.cat(activations, dim=1)

        from sklearn.decomposition import PCA

        pca = PCA(n_components=min(n_concepts, all_activations.shape[1]))
        concepts = pca.fit_transform(all_activations.cpu().numpy())

        return concepts, pca.components_


class TCAV:
    """Testing with Conceptual Sensitivity (TCAV)."""

    def __init__(self, model: nn.Module):
        self.model = model

    def compute_tcav_score(
        self,
        input_examples: Tensor,
        concept_examples: Tensor,
        target_class: int,
        random_examples: Tensor,
        n_derivatives: int = 50,
    ) -> float:
        """
        Compute TCAV score for a concept.

        Args:
            input_examples: Batch of input examples
            concept_examples: Examples representing the concept
            target_class: Target class for the score
            random_examples: Random baseline examples
            n_derivatives: Number of derivatives to compute
        """
        grad_concept = self._compute_directional_derivative(
            input_examples, concept_examples, target_class, n_derivatives
        )

        grad_random = self._compute_directional_derivative(
            input_examples, random_examples, target_class, n_derivatives
        )

        tcav_score = (grad_concept > grad_random).float().mean().item()

        return tcav_score

    def _compute_directional_derivative(
        self,
        inputs: Tensor,
        direction: Tensor,
        target_class: int,
        n_samples: int,
    ) -> Tensor:
        """Compute directional derivative."""
        self.model.eval()

        gradients = []

        for _ in range(n_samples):
            alpha = torch.rand_like(inputs) * 0.1
            perturbed = inputs + alpha * direction

            perturbed.requires_grad = True
            output = self.model(perturbed)

            if target_class >= output.shape[1]:
                target_class = output.argmax(dim=1).item()

            self.model.zero_grad()
            output[0, target_class].backward()

            if perturbed.grad is not None:
                gradients.append(perturbed.grad.clone())

        if not gradients:
            return torch.zeros(inputs.shape[0])

        return torch.stack(gradients).mean(dim=0).mean(dim=0)


class ConceptBottleneck:
    """Concept bottleneck model for interpretable predictions."""

    def __init__(
        self, input_dim: int, n_concepts: int, n_classes: int, hidden_dim: int = 128
    ):
        self.n_concepts = n_concepts

        self.concept_predictor = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, n_concepts),
            nn.Sigmoid(),
        )

        self.class_predictor = nn.Sequential(
            nn.Linear(n_concepts, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, n_classes),
        )

    def forward(self, x: Tensor) -> Tuple[Tensor, Tensor]:
        concepts = self.concept_predictor(x)
        logits = self.class_predictor(concepts)
        return concepts, logits

    def predict_with_concepts(
        self, x: Tensor, concept_names: Optional[List[str]] = None
    ) -> dict:
        """Predict and return concepts for interpretation."""
        concepts, logits = self.forward(x)
        predictions = logits.argmax(dim=-1)

        result = {
            "predictions": predictions,
            "concept_values": concepts,
            "logits": logits,
        }

        if concept_names:
            result["concept_names"] = concept_names

        return result


class LinearProbe:
    """Train linear probes on model representations."""

    def __init__(self, input_dim: int, output_dim: int):
        self.probe = nn.Linear(input_dim, output_dim)
        self.fitted = False

    def fit(
        self,
        representations: Tensor,
        labels: Tensor,
        lr: float = 1e-3,
        epochs: int = 100,
    ):
        """Fit linear probe on representations."""
        optimizer = torch.optim.Adam(self.probe.parameters(), lr=lr)
        criterion = nn.CrossEntropyLoss()

        self.probe.train()
        for _ in range(epochs):
            optimizer.zero_grad()
            logits = self.probe(representations)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()

        self.fitted = True

    def predict(self, representations: Tensor) -> Tensor:
        """Predict using linear probe."""
        if not self.fitted:
            raise ValueError("Probe not fitted yet")
        self.probe.eval()
        with torch.no_grad():
            return self.probe(representations)
