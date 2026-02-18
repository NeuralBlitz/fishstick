"""
Counterfactual Explanations

This module provides counterfactual explanation methods that answer
"What should change to get a different outcome?"

Implements multiple counterfactual generation algorithms:
- GrowingSpheres: Find minimal changes using sphere expansion
- DiCE: Diverse Counterfactual Explanations
- ProtoPF: Prototypical Part-First counterfactuals
- Actionable Counterfactuals: Constrained generation
"""

from __future__ import annotations

from typing import Optional, List, Dict, Tuple, Union, Callable
from dataclasses import dataclass
from abc import ABC, abstractmethod
import copy

import torch
from torch import Tensor, nn
import numpy as np


@dataclass
class Counterfactual:
    """Represents a counterfactual explanation."""

    input: Tensor
    original_prediction: float
    counterfactual_prediction: float
    changes: Dict[str, float]
    distance: float
    is_valid: bool = True


class CounterfactualGenerator(ABC):
    """Abstract base class for counterfactual generators."""

    def __init__(self, model: nn.Module):
        self.model = model
        self.model.eval()

    @abstractmethod
    def generate(
        self,
        input_tensor: Tensor,
        target_class: Optional[int] = None,
        **kwargs,
    ) -> Counterfactual:
        """Generate a counterfactual explanation."""
        pass


class GrowingSpheres(CounterfactualGenerator):
    """Growing Spheres counterfactual generation algorithm.

    Finds the smallest change to the input that flips the prediction
    by iteratively expanding a sphere around the input until a
    counterfactual is found.

    Args:
        model: Model to generate counterfactuals for
        expansion_factor: Factor to expand search radius
        max_iterations: Maximum number of iterations
        target_threshold: Confidence threshold for target class

    Example:
        >>> gs = GrowingSpheres(model)
        >>> cf = gs.generate(input_tensor, target_class=1)
    """

    def __init__(
        self,
        model: nn.Module,
        expansion_factor: float = 1.5,
        max_iterations: int = 100,
        target_threshold: float = 0.5,
    ):
        super().__init__(model)
        self.expansion_factor = expansion_factor
        self.max_iterations = max_iterations
        self.target_threshold = target_threshold

    def generate(
        self,
        input_tensor: Tensor,
        target_class: Optional[int] = None,
        allowed_changes: Optional[Tensor] = None,
    ) -> Counterfactual:
        """Generate counterfactual using Growing Spheres algorithm.

        Args:
            input_tensor: Original input
            target_class: Desired target class (if None, any different class)
            allowed_changes: Binary mask of editable features

        Returns:
            Counterfactual explanation
        """
        original_pred = self._get_prediction(input_tensor)

        radius = 0.1
        best_cf = None
        best_distance = float("inf")

        for iteration in range(self.max_iterations):
            candidates = self._sample_candidates(input_tensor, radius, allowed_changes)

            for candidate in candidates:
                pred = self._get_prediction(candidate)

                if self._is_counterfactual(pred, target_class):
                    distance = self._compute_distance(input_tensor, candidate)

                    if distance < best_distance:
                        best_distance = distance
                        best_cf = candidate

            if best_cf is not None:
                break

            radius *= self.expansion_factor

        if best_cf is None:
            return Counterfactual(
                input=input_tensor,
                original_prediction=original_pred,
                counterfactual_prediction=original_pred,
                changes={},
                distance=float("inf"),
                is_valid=False,
            )

        final_pred = self._get_prediction(best_cf)
        changes = self._compute_changes(input_tensor, best_cf)

        return Counterfactual(
            input=best_cf,
            original_prediction=original_pred,
            counterfactual_prediction=final_pred,
            changes=changes,
            distance=best_distance,
            is_valid=True,
        )

    def _get_prediction(self, inputs: Tensor) -> Tensor:
        """Get model prediction."""
        with torch.no_grad():
            output = self.model(inputs)
            return torch.softmax(output, dim=-1)

    def _is_counterfactual(
        self,
        predictions: Tensor,
        target_class: Optional[int],
    ) -> bool:
        """Check if prediction is a valid counterfactual."""
        if target_class is not None:
            return predictions[0, target_class] > self.target_threshold
        else:
            return predictions.argmax(dim=-1)[0] != predictions.argmax(dim=-1)[0]

    def _sample_candidates(
        self,
        input_tensor: Tensor,
        radius: float,
        allowed_changes: Optional[Tensor],
    ) -> List[Tensor]:
        """Sample counterfactual candidates on sphere surface."""
        n_candidates = 20
        candidates = []

        for _ in range(n_candidates):
            direction = torch.randn_like(input_tensor)
            direction = direction / direction.norm()

            candidate = input_tensor + radius * direction

            if allowed_changes is not None:
                candidate = (
                    input_tensor * (1 - allowed_changes) + candidate * allowed_changes
                )

            candidates.append(candidate)

        return candidates

    def _compute_distance(
        self,
        original: Tensor,
        counterfactual: Tensor,
    ) -> float:
        """Compute distance between inputs."""
        return torch.norm(original - counterfactual).item()

    def _compute_changes(
        self,
        original: Tensor,
        counterfactual: Tensor,
    ) -> Dict[str, float]:
        """Compute feature changes."""
        diff = (counterfactual - original).abs()
        return {"l2_distance": diff.sum().item()}


class DiCEGenerator(CounterfactualGenerator):
    """Diverse Counterfactual Explanations (DiCE).

    Generates multiple diverse counterfactual explanations that show
    different ways to achieve the desired outcome.

    Args:
        model: Model to generate counterfactuals for
        num_cfs: Number of counterfactuals to generate
        diversity_weight: Weight for diversity vs. proximity

    Example:
        >>> dice = DiCEGenerator(model, num_cfs=3)
        >>> cfs = dice.generate(input_tensor, target_class=1)
    """

    def __init__(
        self,
        model: nn.Module,
        num_cfs: int = 3,
        diversity_weight: float = 0.5,
        proximity_weight: float = 0.2,
    ):
        super().__init__(model)
        self.num_cfs = num_cfs
        self.diversity_weight = diversity_weight
        self.proximity_weight = proximity_weight

    def generate(
        self,
        input_tensor: Tensor,
        target_class: Optional[int] = None,
        constraints: Optional[Dict] = None,
    ) -> List[Counterfactual]:
        """Generate diverse counterfactuals.

        Args:
            input_tensor: Original input
            target_class: Target class for counterfactuals
            constraints: Feature constraints

        Returns:
            List of counterfactual explanations
        """
        original_pred = self._get_prediction(input_tensor)
        target_class = target_class or self._get_default_target(original_pred)

        cfs = []
        for i in range(self.num_cfs):
            cf = self._generate_single_cf(
                input_tensor, target_class, constraints, seed=i
            )
            cfs.append(cf)

        return cfs

    def _generate_single_cf(
        self,
        input_tensor: Tensor,
        target_class: int,
        constraints: Optional[Dict],
        seed: int,
    ) -> Counterfactual:
        """Generate a single counterfactual with diversity."""
        torch.manual_seed(seed)
        np.random.seed(seed)

        cf = input_tensor.clone()

        optimizer = torch.optim.Adam([cf.requires_grad_()], lr=0.01)

        for step in range(200):
            optimizer.zero_grad()

            proximity_loss = -self.proximity_weight * self._proximity_loss(
                input_tensor, cf
            )

            if len(self._counterfactuals) > 0 and self.diversity_weight > 0:
                diversity_loss = self.diversity_weight * self._diversity_loss(
                    cf, self._counterfactuals
                )
            else:
                diversity_loss = 0

            output = self.model(cf)
            target_prob = output[0, target_class]

            if target_prob > 0.5:
                conf_loss = -torch.log(target_prob + 1e-8)
            else:
                conf_loss = -torch.log(1 - target_prob + 1e-8)

            if constraints:
                constraint_loss = self._constraint_loss(cf, constraints)
            else:
                constraint_loss = 0

            loss = conf_loss + proximity_loss + diversity_loss + constraint_loss

            if conf_loss.item() < 0.5:
                break

            loss.backward()
            optimizer.step()

        self._counterfactuals.append(cf.detach())

        final_pred = self._get_prediction(cf)
        changes = self._compute_changes(input_tensor, cf)

        return Counterfactual(
            input=cf.detach(),
            original_prediction=self._get_prediction(input_tensor).max().item(),
            counterfactual_prediction=final_pred.max().item(),
            changes=changes,
            distance=self._compute_distance(input_tensor, cf),
            is_valid=True,
        )

    def _get_prediction(self, inputs: Tensor) -> Tensor:
        """Get model predictions."""
        with torch.no_grad():
            output = self.model(inputs)
            return torch.softmax(output, dim=-1)

    def _get_default_target(self, predictions: Tensor) -> int:
        """Get default target class (least likely non-original)."""
        probs = predictions[0]
        sorted_probs, indices = probs.sort()
        return indices[0].item()

    def _proximity_loss(self, original: Tensor, cf: Tensor) -> Tensor:
        """Compute proximity (similarity) to original input."""
        return -torch.norm(original - cf)

    def _diversity_loss(self, cf: Tensor, existing_cfs: List[Tensor]) -> Tensor:
        """Compute diversity loss for multiple counterfactuals."""
        if not existing_cfs:
            return torch.tensor(0.0)

        existing = torch.stack([c.detach() for c in existing_cfs])
        distances = torch.cdist(cf.unsqueeze(0), existing).squeeze()

        return -distances.mean()

    def _constraint_loss(
        self,
        cf: Tensor,
        constraints: Dict,
    ) -> Tensor:
        """Compute loss from user-defined constraints."""
        loss = torch.tensor(0.0)

        if "feature_bounds" in constraints:
            bounds = constraints["feature_bounds"]
            for i, (low, high) in enumerate(bounds):
                if i < cf.numel():
                    idx = np.unravel_index(i, cf.shape)
                    if cf[idx] < low:
                        loss += (low - cf[idx]) ** 2
                    elif cf[idx] > high:
                        loss += (cf[idx] - high) ** 2

        return loss

    def _compute_distance(
        self,
        original: Tensor,
        counterfactual: Tensor,
    ) -> float:
        """Compute distance between inputs."""
        return torch.norm(original - counterfactual).item()

    def _compute_changes(
        self,
        original: Tensor,
        counterfactual: Tensor,
    ) -> Dict[str, float]:
        """Compute feature changes."""
        diff = (counterfactual - original).abs()
        return {"l2_distance": diff.sum().item()}


class ProtoPFGenerator(CounterfactualGenerator):
    """Prototypical Part-First counterfactual generation.

    Generates counterfactuals by finding prototype parts from other classes
    and modifying the input towards those prototypes.

    Args:
        model: Model to explain
        prototypes: Prototype embeddings for each class

    Example:
        >>> proto = ProtoPFGenerator(model, prototypes_dict)
        >>> cf = proto.generate(input_tensor, target_class=1)
    """

    def __init__(
        self,
        model: nn.Module,
        prototypes: Optional[Dict[int, Tensor]] = None,
    ):
        super().__init__(model)
        self.prototypes = prototypes or {}

    def set_prototypes(
        self,
        prototypes: Dict[int, Tensor],
    ):
        """Set prototype embeddings for each class."""
        self.prototypes = prototypes

    def generate(
        self,
        input_tensor: Tensor,
        target_class: Optional[int] = None,
        num_steps: int = 50,
    ) -> Counterfactual:
        """Generate counterfactual using prototypes.

        Args:
            input_tensor: Original input
            target_class: Target class (if None, selects least likely class)
            num_steps: Number of optimization steps

        Returns:
            Counterfactual explanation
        """
        original_pred = self._get_prediction(input_tensor)

        if target_class is None:
            target_class = self._get_default_target(original_pred)

        if target_class not in self.prototypes:
            return self._generate_fallback(input_tensor, target_class)

        prototype = self.prototypes[target_class].to(input_tensor.device)

        if prototype.dim() != input_tensor.dim():
            prototype = self._resize_prototype(prototype, input_tensor.shape)

        cf = input_tensor.clone().requires_grad_(True)
        optimizer = torch.optim.Adam([cf], lr=0.01)

        original_pred_class = original_pred.argmax(dim=-1).item()

        for step in range(num_steps):
            optimizer.zero_grad()

            output = self.model(cf)
            target_prob = output[0, target_class]

            prototype_loss = torch.norm(cf - prototype)

            if step > num_steps // 2:
                loss = -torch.log(target_prob + 1e-8) + 0.1 * prototype_loss
            else:
                loss = prototype_loss

            loss.backward()
            optimizer.step()

            if target_prob.item() > 0.5:
                break

        final_pred = self._get_prediction(cf)
        changes = self._compute_changes(input_tensor, cf)

        return Counterfactual(
            input=cf.detach(),
            original_prediction=original_pred.max().item(),
            counterfactual_prediction=final_pred.max().item(),
            changes=changes,
            distance=self._compute_distance(input_tensor, cf),
            is_valid=target_class == final_pred.argmax(dim=-1).item(),
        )

    def _generate_fallback(
        self,
        input_tensor: Tensor,
        target_class: int,
    ) -> Counterfactual:
        """Fallback generation when no prototypes available."""
        cf = input_tensor.clone() + torch.randn_like(input_tensor) * 0.5
        final_pred = self._get_prediction(cf)

        return Counterfactual(
            input=cf,
            original_prediction=self._get_prediction(input_tensor).max().item(),
            counterfactual_prediction=final_pred.max().item(),
            changes={"note": "fallback_generation"},
            distance=float("inf"),
            is_valid=False,
        )

    def _resize_prototype(
        self,
        prototype: Tensor,
        target_shape: Tuple,
    ) -> Tensor:
        """Resize prototype to match input shape."""
        if prototype.dim() == 1:
            return prototype.view(1, -1, 1, 1).expand(target_shape)
        return prototype

    def _get_prediction(self, inputs: Tensor) -> Tensor:
        """Get model predictions."""
        with torch.no_grad():
            output = self.model(inputs)
            return torch.softmax(output, dim=-1)

    def _get_default_target(self, predictions: Tensor) -> int:
        """Get default target class."""
        return predictions[0].argmin().item()

    def _compute_distance(
        self,
        original: Tensor,
        counterfactual: Tensor,
    ) -> float:
        """Compute distance between inputs."""
        return torch.norm(original - counterfactual).item()

    def _compute_changes(
        self,
        original: Tensor,
        counterfactual: Tensor,
    ) -> Dict[str, float]:
        """Compute feature changes."""
        diff = (counterfactual - original).abs()
        return {"l2_distance": diff.sum().item()}


class ActionableCounterfactual(CounterfactualGenerator):
    """Actionable counterfactual generation with constraints.

    Generates counterfactuals that are:
    - Actionable: Changes can be realistically made
    - Plausible: Changes are realistic
    - Minimal: Changes are as small as possible

    Args:
        model: Model to explain
        feature_ranges: Allowed ranges for each feature

    Example:
        >>> ac = ActionableCounterfactual(model, feature_ranges={'age': (0, 100)})
        >>> cf = ac.generate(input_tensor, target_class=1)
    """

    def __init__(
        self,
        model: nn.Module,
        feature_ranges: Optional[Dict[str, Tuple[float, float]]] = None,
        actionability_weight: float = 0.5,
    ):
        super().__init__(model)
        self.feature_ranges = feature_ranges or {}
        self.actionability_weight = actionability_weight

    def generate(
        self,
        input_tensor: Tensor,
        target_class: Optional[int] = None,
        immutable_features: Optional[List[int]] = None,
    ) -> Counterfactual:
        """Generate actionable counterfactual.

        Args:
            input_tensor: Original input
            target_class: Target class
            immutable_features: Indices of features that cannot change

        Returns:
            Counterfactual explanation
        """
        original_pred = self._get_prediction(input_tensor)

        if target_class is None:
            target_class = self._get_default_target(original_pred)

        immutable_features = immutable_features or []

        cf = input_tensor.clone().requires_grad_(True)
        optimizer = torch.optim.Adam([cf], lr=0.01)

        for step in range(300):
            optimizer.zero_grad()

            output = self.model(cf)
            target_prob = output[0, target_class]

            proximity_loss = torch.norm(input_tensor - cf)

            actionability_loss = self._actionability_loss(cf, immutable_features)

            if target_prob.item() > 0.5:
                loss = proximity_loss + self.actionability_weight * actionability_loss
            else:
                loss = -torch.log(target_prob + 1e-8) + proximity_loss

            loss.backward()
            optimizer.step()

            self._apply_constraints(cf)

            if target_prob.item() > 0.7:
                break

        final_pred = self._get_prediction(cf)
        changes = self._compute_changes(input_tensor, cf)

        return Counterfactual(
            input=cf.detach(),
            original_prediction=original_pred.max().item(),
            counterfactual_prediction=final_pred.max().item(),
            changes=changes,
            distance=self._compute_distance(input_tensor, cf),
            is_valid=target_class == final_pred.argmax(dim=-1).item(),
        )

    def _actionability_loss(
        self,
        cf: Tensor,
        immutable_features: List[int],
    ) -> Tensor:
        """Compute actionability loss (changes to mutable features only)."""
        loss = torch.tensor(0.0)

        for idx in immutable_features:
            if idx < cf.numel():
                orig_val = cf.flatten()[idx].detach()
                loss += (cf.flatten()[idx] - orig_val) ** 2

        return loss

    def _apply_constraints(self, cf: Tensor):
        """Apply feature constraints to counterfactual."""
        if not self.feature_ranges:
            return

        for i, (feature_name, (low, high)) in enumerate(self.feature_ranges.items()):
            if i < cf.numel():
                cf.flatten()[i] = torch.clamp(cf.flatten()[i], min=low, max=high)

    def _get_prediction(self, inputs: Tensor) -> Tensor:
        """Get model predictions."""
        with torch.no_grad():
            output = self.model(inputs)
            return torch.softmax(output, dim=-1)

    def _get_default_target(self, predictions: Tensor) -> int:
        """Get default target class."""
        sorted_probs, indices = predictions[0].sort()
        for idx in indices:
            if idx.item() != predictions[0].argmax().item():
                return idx.item()
        return 0

    def _compute_distance(
        self,
        original: Tensor,
        counterfactual: Tensor,
    ) -> float:
        """Compute distance between inputs."""
        return torch.norm(original - counterfactual).item()

    def _compute_changes(
        self,
        original: Tensor,
        counterfactual: Tensor,
    ) -> Dict[str, float]:
        """Compute feature changes."""
        diff = (counterfactual - original).abs()
        return {"l2_distance": diff.sum().item()}


def create_counterfactual_generator(
    model: nn.Module,
    method: str = "growing_spheres",
    **kwargs,
) -> CounterfactualGenerator:
    """Factory function to create counterfactual generators.

    Args:
        model: Model to generate counterfactuals for
        method: Generation method ('growing_spheres', 'dice', 'proto', 'actionable')
        **kwargs: Additional arguments

    Returns:
        Configured counterfactual generator

    Example:
        >>> gen = create_counterfactual_generator(model, 'dice', num_cfs=3)
    """
    if method == "growing_spheres":
        return GrowingSpheres(model, **kwargs)
    elif method == "dice":
        return DiCEGenerator(model, **kwargs)
    elif method == "proto":
        return ProtoPFGenerator(model, **kwargs)
    elif method == "actionable":
        return ActionableCounterfactual(model, **kwargs)
    else:
        raise ValueError(f"Unknown method: {method}")
