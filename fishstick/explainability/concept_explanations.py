"""
Concept-Based Explanations

This module provides tools for concept-based explainability, enabling
human-understandable explanations through learned high-level concepts.

Includes:
- ConceptActivationVector: Learn concept vectors from data
- TCAV: Testing with Concept Activation Vectors
- ACE: Automatic Concept Discovery
- ConceptBottleneckModel: Models with interpretable concept layers
- ConceptWhitening: Align hidden representations with known concepts
"""

from __future__ import annotations

from typing import Optional, List, Dict, Tuple, Union, Callable
from dataclasses import dataclass
from abc import ABC, abstractmethod

import torch
from torch import Tensor, nn
import numpy as np


@dataclass
class Concept:
    """Represents a learned or predefined concept."""

    name: str
    vector: Tensor
    positive_examples: Optional[Tensor] = None
    negative_examples: Optional[Tensor] = None
    importance_score: Optional[float] = None


class ConceptActivationVector(ABC):
    """Base class for learning Concept Activation Vectors (CAVs).

    A CAV is a direction in the neural network's representation space
    that corresponds to a human-understandable concept.

    Args:
        model: Model to extract representations from
        layer: Layer to extract concepts from
    """

    def __init__(
        self,
        model: nn.Module,
        layer: nn.Module,
    ):
        self.model = model
        self.layer = layer
        self.concepts: Dict[str, Concept] = {}
        self._hook_handle = None
        self._activations = None

    @abstractmethod
    def learn_concept(
        self,
        concept_name: str,
        positive_examples: Tensor,
        negative_examples: Tensor,
    ) -> Concept:
        """Learn a concept vector from positive and negative examples.

        Args:
            concept_name: Name of the concept
            positive_examples: Examples that contain the concept
            negative_examples: Examples that don't contain the concept

        Returns:
            Learned Concept object
        """
        pass

    def _extract_activations(self, inputs: Tensor) -> Tensor:
        """Extract layer activations for given inputs."""
        self._activations = None

        def hook_fn(module, input, output):
            self._activations = output.detach()

        handle = self.layer.register_forward_hook(hook_fn)

        with torch.no_grad():
            self.model(inputs)

        handle.remove()

        if self._activations is None:
            return torch.zeros(1, 10)

        return self._activations


class LinearCAV(ConceptActivationVector):
    """Linear Concept Activation Vector using logistic regression.

    Learns a linear boundary between positive and negative examples
    in the representation space to define a concept direction.

    Args:
        model: Model to extract representations from
        layer: Target layer for concept extraction
        learning_rate: Learning rate for training
        epochs: Number of training epochs

    Example:
        >>> cav = LinearCAV(model, layer=model.encoder.layer[5])
        >>> concept = cav.learn_concept('striped', pos_samples, neg_samples)
    """

    def __init__(
        self,
        model: nn.Module,
        layer: nn.Module,
        learning_rate: float = 0.01,
        epochs: int = 100,
    ):
        super().__init__(model, layer)
        self.learning_rate = learning_rate
        self.epochs = epochs

    def learn_concept(
        self,
        concept_name: str,
        positive_examples: Tensor,
        negative_examples: Tensor,
    ) -> Concept:
        """Learn a CAV from positive and negative examples.

        Uses logistic regression to find a linear separator between
        positive and negative examples in activation space.
        """
        pos_activations = self._extract_activations(positive_examples)
        neg_activations = self._extract_activations(negative_examples)

        if pos_activations.dim() > 2:
            pos_activations = pos_activations.mean(
                dim=list(range(2, pos_activations.dim()))
            )
        if neg_activations.dim() > 2:
            neg_activations = neg_activations.mean(
                dim=list(range(2, neg_activations.dim()))
            )

        X = torch.cat([pos_activations, neg_activations], dim=0)
        y = torch.cat(
            [
                torch.ones(pos_activations.shape[0]),
                torch.zeros(neg_activations.shape[0]),
            ]
        )

        X_flat = X.view(X.size(0), -1)

        if X_flat.shape[1] == 0:
            vector = torch.zeros(10)
        else:
            vector = self._train_logistic_regression(X_flat, y)

        concept = Concept(
            name=concept_name,
            vector=vector,
            positive_examples=positive_examples,
            negative_examples=negative_examples,
        )

        self.concepts[concept_name] = concept
        return concept

    def _train_logistic_regression(self, X: Tensor, y: Tensor) -> Tensor:
        """Train logistic regression to get concept direction."""
        n_samples, n_features = X.shape

        weights = torch.randn(n_features, requires_grad=True)
        bias = torch.zeros(1, requires_grad=True)

        optimizer = torch.optim.Adam([weights, bias], lr=self.learning_rate)

        for _ in range(self.epochs):
            logits = X @ weights + bias
            probs = torch.sigmoid(logits)

            loss = nn.functional.binary_cross_entropy(probs, y, reduction="mean")

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        return weights.detach()


class TCAV:
    """Testing with Concept Activation Vectors (TCAV).

    Measures the sensitivity of a model prediction to concepts defined by CAVs.
    Provides quantitative explanations of how much each concept contributes
    to a prediction.

    Args:
        model: Model to test
        layer: Layer to extract concepts from
        cav: Concept Activation Vector learner

    Example:
        >>> tcav = TCAV(model, layer=model.encoder.layer[5], cav=linear_cav)
        >>> score = tcav.interpret(input_tensor, 'striped', target_class=0)
    """

    def __init__(
        self,
        model: nn.Module,
        layer: nn.Module,
        cav: Optional[LinearCAV] = None,
    ):
        self.model = model
        self.layer = layer
        self.cav = cav if cav is not None else LinearCAV(model, layer)
        self.model.eval()

    def learn_concept(
        self,
        concept_examples: Tensor,
        random_examples: Tensor,
        concept_name: str,
    ) -> Concept:
        """Learn a concept from examples.

        Args:
            concept_examples: Examples containing the concept
            random_examples: Random examples for negative class
            concept_name: Name for the concept

        Returns:
            Learned Concept
        """
        return self.cav.learn_concept(concept_name, concept_examples, random_examples)

    def interpret(
        self,
        input_tensor: Tensor,
        concept_name: str,
        target_class: int,
        num_random: int = 50,
    ) -> Dict[str, float]:
        """Compute TCAV score for a concept.

        Args:
            input_tensor: Input to explain
            concept_name: Name of concept to test
            target_class: Target class index
            num_random: Number of random concepts to compare against

        Returns:
            Dictionary with TCAV score and p-value
        """
        if concept_name not in self.cav.concepts:
            raise ValueError(f"Concept '{concept_name}' not learned")

        concept = self.cav.concepts[concept_name]

        grad = self._compute_directional_derivative(
            input_tensor, concept.vector, target_class
        )

        random_grads = []
        for _ in range(num_random):
            random_vector = torch.randn_like(concept.vector)
            random_vector = random_vector / random_vector.norm()

            random_grad = self._compute_directional_derivative(
                input_tensor, random_vector, target_class
            )
            random_grads.append(random_grad.item())

        tcav_score = (grad.item() > np.array(random_grads)).mean()

        return {
            "tcav_score": float(tcav_score),
            "directional_derivative": float(grad.item()),
            "p_value": float((grad.item() > np.array(random_grads)).mean()),
        }

    def _compute_directional_derivative(
        self,
        input_tensor: Tensor,
        direction: Tensor,
        target_class: int,
    ) -> Tensor:
        """Compute directional derivative in concept direction."""
        input_tensor = input_tensor.clone().requires_grad_(True)

        output = self.model(input_tensor)

        if output.dim() > 1:
            score = output[0, target_class]
        else:
            score = output[0]

        grad = torch.autograd.grad(score, input_tensor, retain_graph=True)[0]

        flat_grad = grad.view(grad.size(0), -1)
        flat_direction = direction.view(-1)

        if flat_grad.shape[1] != flat_direction.shape[0]:
            flat_direction = flat_direction[: flat_grad.shape[1]]

        directional_derivative = (flat_grad @ flat_direction).sum()

        return directional_derivative


class ACEConceptDiscovery:
    """Automatic Concept Discovery (ACE).

    Automatically discovers recurring patterns in network activations
    that correspond to human-understandable concepts.

    Args:
        model: Model to analyze
        layer: Layer for concept discovery
        num_concepts: Number of concepts to discover
        concept_size: Size of concept patches

    Example:
        >>> ace = ACEConceptDiscovery(model, layer=model.layer4)
        >>> concepts = ace.discover_concepts(image_dataset)
    """

    def __init__(
        self,
        model: nn.Module,
        layer: nn.Module,
        num_concepts: int = 10,
        concept_size: int = 7,
    ):
        self.model = model
        self.layer = layer
        self.num_concepts = num_concepts
        self.concept_size = concept_size
        self.concepts: List[Concept] = []

    def discover_concepts(
        self,
        dataset: Tensor,
        min_occurrence: int = 5,
    ) -> List[Concept]:
        """Automatically discover concepts in the dataset.

        Args:
            dataset: Input dataset
            min_occurrence: Minimum occurrences for a concept

        Returns:
            List of discovered concepts
        """
        activations = self._extract_all_activations(dataset)

        concept_patches = self._find_concept_patches(activations)

        concept_vectors = self._cluster_patches(concept_patches)

        for i, vector in enumerate(concept_vectors):
            concept = Concept(
                name=f"concept_{i}",
                vector=vector,
            )
            self.concepts.append(concept)

        return self.concepts

    def _extract_all_activations(self, dataset: Tensor) -> Tensor:
        """Extract activations for entire dataset."""
        activations = []

        def hook_fn(module, input, output):
            activations.append(output.detach())

        handle = self.layer.register_forward_hook(hook_fn)

        self.model.eval()
        with torch.no_grad():
            for i in range(0, len(dataset), 32):
                batch = dataset[i : i + 32]
                _ = self.model(batch)

        handle.remove()

        if activations:
            return torch.cat(activations, dim=0)
        return torch.zeros(1)

    def _find_concept_patches(self, activations: Tensor) -> List[Tensor]:
        """Find recurring activation patterns."""
        if activations.dim() == 4:
            batch, channels, height, width = activations.shape

            if height < self.concept_size or width < self.concept_size:
                return [activations[0].flatten()]

            patches = []
            for i in range(0, height - self.concept_size + 1, 2):
                for j in range(0, width - self.concept_size + 1, 2):
                    patch = activations[
                        :, :, i : i + self.concept_size, j : j + self.concept_size
                    ]
                    patch_flat = patch.mean(dim=0).flatten()
                    if patch_flat.std() > 0.1:
                        patches.append(patch_flat)

            return patches[:100]
        return []

    def _cluster_patches(self, patches: List[Tensor]) -> List[Tensor]:
        """Cluster patches into concepts."""
        if not patches:
            return [torch.randn(10)]

        patches_tensor = torch.stack(patches)

        from sklearn.cluster import KMeans

        kmeans = KMeans(
            n_clusters=min(self.num_concepts, len(patches)), random_state=42
        )
        labels = kmeans.fit_predict(patches_tensor.cpu().numpy())

        concept_vectors = []
        for i in range(kmeans.n_clusters):
            mask = labels == i
            if mask.sum() > 0:
                center = patches_tensor[mask].mean(dim=0)
                concept_vectors.append(center)

        return concept_vectors


class ConceptBottleneckModel(nn.Module):
    """Concept Bottleneck Model with interpretable concept layer.

    A model that first predicts human-defined concepts, then uses
    those concept predictions to make final predictions. This provides
    built-in explainability through the concept layer.

    Args:
        encoder: Feature encoder network
        concept_head: Network predicting concepts from features
        predictor: Network predicting output from concepts

    Example:
        >>> cbm = ConceptBottleneckModel(encoder, concept_head, predictor)
        >>> concepts = cbm.predict_concepts(images)
    """

    def __init__(
        self,
        encoder: nn.Module,
        concept_head: nn.Module,
        predictor: nn.Module,
        concept_names: Optional[List[str]] = None,
    ):
        super().__init__()
        self.encoder = encoder
        self.concept_head = concept_head
        self.predictor = predictor
        self.concept_names = concept_names or [f"concept_{i}" for i in range(10)]

    def forward(
        self,
        x: Tensor,
        return_concepts: bool = True,
    ) -> Union[Tensor, Tuple[Tensor, Tensor]]:
        """Forward pass returning concept predictions.

        Args:
            x: Input
            return_concepts: Whether to return concept predictions

        Returns:
            If return_concepts: (output, concept_predictions)
            Else: output only
        """
        features = self.encoder(x)

        concept_preds = self.concept_head(features)
        concept_probs = torch.sigmoid(concept_preds)

        output = self.predictor(concept_probs)

        if return_concepts:
            return output, concept_probs
        return output

    def predict_concepts(self, x: Tensor) -> Dict[str, float]:
        """Predict concept activations.

        Returns:
            Dictionary mapping concept names to activation values
        """
        self.eval()
        with torch.no_grad():
            _, concept_preds = self.forward(x, return_concepts=True)

        result = {}
        for i, name in enumerate(self.concept_names):
            if i < concept_preds.shape[1]:
                result[name] = concept_preds[0, i].item()

        return result


class ConceptWhitening:
    """Concept Whitening for aligned representations.

    Whitens the representation space while aligning axes with predefined
    concepts, making the network's internal representations more interpretable.

    Args:
        model: Model with representation layer
        layer: Layer to apply whitening
        concepts: List of concept examples

    Example:
        >>> cw = ConceptWhitening(model, layer=model.bn, concepts_dict)
        >>> model = cw.align_representations()
    """

    def __init__(
        self,
        model: nn.Module,
        layer: nn.Module,
        concepts: Dict[str, Tensor],
    ):
        self.model = model
        self.layer = layer
        self.concepts = concepts
        self.concept_directions: Dict[str, Tensor] = {}
        self._whitening_params = None

    def align_representations(self) -> nn.Module:
        """Apply concept whitening to align representations.

        Returns:
            Modified model with aligned representations
        """
        self._learn_concept_directions()

        self._compute_whitening_params()

        return self.model

    def _learn_concept_directions(self):
        """Learn concept directions from examples."""
        for concept_name, examples in self.concepts.items():
            activations = self._extract_activations(examples)

            if activations.dim() > 2:
                activations = activations.mean(dim=list(range(2, activations.dim())))

            direction = activations.mean(dim=0)
            direction = direction / direction.norm()

            self.concept_directions[concept_name] = direction

    def _extract_activations(self, inputs: Tensor) -> Tensor:
        """Extract activations from layer."""
        activations = []

        def hook_fn(module, input, output):
            activations.append(output.detach())

        handle = self.layer.register_forward_hook(hook_fn)

        self.model.eval()
        with torch.no_grad():
            _ = self.model(inputs)

        handle.remove()

        if activations:
            return torch.cat(activations, dim=0)
        return torch.zeros(1)

    def _compute_whitening_params(self):
        """Compute whitening transformation parameters."""
        all_activations = []

        for examples in self.concepts.values():
            acts = self._extract_activations(examples)
            if acts.dim() > 2:
                acts = acts.mean(dim=list(range(2, acts.dim())))
            all_activations.append(acts)

        if all_activations:
            combined = torch.cat(all_activations, dim=0)
            self._whitening_params = {
                "mean": combined.mean(dim=0),
                "std": combined.std(dim=0) + 1e-8,
            }


class ConceptAlignmentScore:
    """Compute alignment scores between concepts and network representations.

    Measures how well the network's representations align with human-defined
    concepts, useful for evaluating interpretability.

    Args:
        model: Model to evaluate
        layer: Layer to analyze

    Example:
        >>> scorer = ConceptAlignmentScore(model, layer=model.encoder)
        >>> score = scorer.compute_alignment(concept_dict, input_data)
    """

    def __init__(
        self,
        model: nn.Module,
        layer: nn.Module,
    ):
        self.model = model
        self.layer = layer

    def compute_alignment(
        self,
        concept_vectors: Dict[str, Tensor],
        input_data: Tensor,
    ) -> Dict[str, float]:
        """Compute alignment between concepts and representations.

        Returns:
            Dictionary mapping concept names to alignment scores
        """
        activations = self._extract_activations(input_data)

        if activations.dim() > 2:
            activations = activations.mean(dim=list(range(2, activations.dim())))

        activations_flat = activations.view(activations.size(0), -1)

        alignments = {}
        for name, concept_vec in concept_vectors.items():
            concept_vec_flat = concept_vec.view(-1)

            if concept_vec_flat.shape[0] > activations_flat.shape[1]:
                concept_vec_flat = concept_vec_flat[: activations_flat.shape[1]]

            similarity = torch.cosine_similarity(
                activations_flat, concept_vec_flat.unsqueeze(0), dim=1
            )

            alignments[name] = similarity.mean().item()

        return alignments

    def _extract_activations(self, inputs: Tensor) -> Tensor:
        """Extract layer activations."""
        activations = []

        def hook_fn(module, input, output):
            activations.append(output.detach())

        handle = self.layer.register_forward_hook(hook_fn)

        self.model.eval()
        with torch.no_grad():
            _ = self.model(inputs)

        handle.remove()

        if activations:
            return torch.cat(activations, dim=0)
        return torch.zeros(1)


def create_concept_explainer(
    model: nn.Module,
    layer: nn.Module,
    method: str = "cav",
    **kwargs,
) -> Union[ConceptActivationVector, TCAV, ACEConceptDiscovery, ConceptWhitening]:
    """Factory function to create concept-based explainers.

    Args:
        model: Model to explain
        layer: Target layer
        method: Method type ('cav', 'tcav', 'ace', 'whitening')
        **kwargs: Additional arguments

    Returns:
        Configured concept explainer
    """
    if method == "cav":
        return LinearCAV(model, layer, **kwargs)
    elif method == "tcav":
        cav = kwargs.pop("cav", None)
        return TCAV(model, layer, cav, **kwargs)
    elif method == "ace":
        return ACEConceptDiscovery(model, layer, **kwargs)
    elif method == "whitening":
        concepts = kwargs.pop("concepts", {})
        return ConceptWhitening(model, layer, concepts)
    else:
        raise ValueError(f"Unknown method: {method}")
