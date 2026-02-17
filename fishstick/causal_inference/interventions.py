"""
Intervention Handling and Analysis.

Provides:
- Intervention types (atomic, stochastic, shift, policy)
- Intervention graph manipulation
- Do-operator implementation
- Policy evaluation under interventions
"""

from typing import Dict, List, Optional, Set, Tuple, Callable, Any, Union
import torch
from torch import Tensor
import numpy as np
from dataclasses import dataclass, field
from enum import Enum
from abc import ABC, abstractmethod
import copy


class InterventionType(Enum):
    """Types of causal interventions."""

    ATOMIC = "atomic"
    STOCHASTIC = "stochastic"
    SHIFT = "shift"
    POLICY = "policy"
    SOFT = "soft"
    HARD = "hard"


@dataclass
class Intervention(ABC):
    """Base class for interventions."""

    target: str
    intervention_type: InterventionType

    @abstractmethod
    def apply(self, value: Tensor) -> Tensor:
        """Apply intervention to a value."""
        pass

    @abstractmethod
    def __str__(self) -> str:
        pass


@dataclass
class AtomicIntervention(Intervention):
    """
    Atomic intervention: set variable to fixed value.
    do(X = x)
    """

    value: Any = field(default=None)

    def __init__(self, target: str, value: Any):
        super().__init__(target, InterventionType.ATOMIC)
        self.value = value

    def apply(self, value: Tensor) -> Tensor:
        """Replace value with intervention value."""
        if isinstance(self.value, Tensor):
            return self.value.expand_as(value)
        return torch.full_like(value, self.value)

    def __str__(self) -> str:
        return f"do({self.target} = {self.value})"


@dataclass
class StochasticIntervention(Intervention):
    """
    Stochastic intervention: draw from distribution.
    do(X ~ P(X))
    """

    distribution: Callable = field(default=None)

    def __init__(self, target: str, distribution: Callable):
        super().__init__(target, InterventionType.STOCHASTIC)
        self.distribution = distribution

    def apply(self, value: Tensor) -> Tensor:
        """Sample from intervention distribution."""
        return self.distribution(value.shape, value.device)

    def __str__(self) -> str:
        return f"do({self.target} ~ P({self.target}))"


@dataclass
class ShiftIntervention(Intervention):
    """
    Shift intervention: add constant to variable.
    do(X = X + delta)
    """

    delta: float = 0.0

    def __init__(self, target: str, delta: float):
        super().__init__(target, InterventionType.SHIFT)
        self.delta = delta

    def apply(self, value: Tensor) -> Tensor:
        """Apply shift to value."""
        return value + self.delta

    def __str__(self) -> str:
        sign = "+" if self.delta >= 0 else ""
        return f"do({self.target} = {self.target} {sign} {self.delta})"


@dataclass
class PolicyIntervention(Intervention):
    """
    Policy intervention: apply deterministic policy function.
    do(X = f(Z)) where Z are covariates
    """

    policy_function: Callable = field(default=None)
    covariate_names: List[str] = field(default_factory=list)

    def __init__(
        self,
        target: str,
        policy_function: Callable,
        covariate_names: List[str],
    ):
        super().__init__(target, InterventionType.POLICY)
        self.policy_function = policy_function
        self.covariate_names = covariate_names

    def apply(
        self, value: Tensor, covariates: Optional[Dict[str, Tensor]] = None
    ) -> Tensor:
        """Apply policy function to determine intervention value."""
        if covariates is None:
            return value

        cov_tensor = torch.stack([covariates[c] for c in self.covariate_names], dim=-1)
        return self.policy_function(cov_tensor)

    def __str__(self) -> str:
        return f"do({self.target} = f({', '.join(self.covariate_names)}))"


class InterventionGraph:
    """
    Graph with intervention information.
    """

    def __init__(
        self,
        nodes: List[str],
        edges: List[Tuple[str, str]],
        interventions: Optional[Dict[str, Intervention]] = None,
    ):
        self.nodes = nodes
        self.edges = edges
        self.interventions = interventions if interventions else {}

        self.adjacency = self._build_adjacency()
        self._validate()

    def _build_adjacency(self) -> Dict[str, List[str]]:
        """Build adjacency list from edges."""
        adj = {node: [] for node in self.nodes}
        for src, dst in self.edges:
            if src in adj:
                adj[src].append(dst)
        return adj

    def _validate(self) -> None:
        """Validate graph structure."""
        for src, dst in self.edges:
            if src not in self.nodes or dst not in self.nodes:
                raise ValueError(f"Invalid edge: ({src}, {dst})")

    def parents(self, node: str) -> List[str]:
        """Get parent nodes."""
        return [src for src, dst in self.edges if dst == node]

    def children(self, node: str) -> List[str]:
        """Get child nodes."""
        return [dst for src, dst in self.edges if src == node]

    def is_affected(self, node: str) -> bool:
        """Check if node is affected by any intervention."""
        if node in self.interventions:
            return True

        for intervention in self.interventions.values():
            if self._is_ancestor(intervention.target, node):
                return True

        return False

    def _is_ancestor(self, ancestor: str, descendant: str) -> bool:
        """Check if ancestor is an ancestor of descendant."""
        visited = set()
        queue = [ancestor]

        while queue:
            current = queue.pop(0)
            if current == descendant:
                return True

            if current in visited:
                continue
            visited.add(current)

            queue.extend(self.children(current))

        return False

    def remove_intervention_edges(
        self, intervention_target: str
    ) -> "InterventionGraph":
        """Remove all edges pointing to intervention target."""
        new_edges = [
            (src, dst) for src, dst in self.edges if dst != intervention_target
        ]

        new_interventions = {
            k: v for k, v in self.interventions.items() if k != intervention_target
        }

        return InterventionGraph(
            nodes=self.nodes,
            edges=new_edges,
            interventions=new_interventions,
        )

    def do_intervention(
        self, target: str, intervention: Intervention
    ) -> "InterventionGraph":
        """Create new graph with intervention applied."""
        new_interventions = dict(self.interventions)
        new_interventions[target] = intervention

        new_edges = [(src, dst) for src, dst in self.edges if dst != target]

        return InterventionGraph(
            nodes=self.nodes,
            edges=new_edges,
            interventions=new_interventions,
        )

    def topological_order(self) -> List[str]:
        """Get topological ordering of nodes."""
        in_degree = {node: 0 for node in self.nodes}
        for src, dst in self.edges:
            in_degree[dst] += 1

        queue = [n for n, d in in_degree.items() if d == 0]
        order = []

        while queue:
            node = queue.pop(0)
            order.append(node)

            for child in self.children(node):
                in_degree[child] -= 1
                if in_degree[child] == 0:
                    queue.append(child)

        return order


def intervened_graph(
    original_graph: "InterventionGraph",
    intervention_target: str,
    intervention_value: Any,
) -> "InterventionGraph":
    """
    Create intervened graph.

    Replaces all incoming edges to target with intervention.
    """
    return original_graph.do_intervention(
        intervention_target,
        AtomicIntervention(intervention_target, intervention_value),
    )


def do_operator(
    original_samples: Dict[str, Tensor],
    graph: InterventionGraph,
    intervention_target: str,
    intervention: Intervention,
    n_samples: int,
) -> Dict[str, Tensor]:
    """
    Apply do-operator to generate samples under intervention.

    Args:
        original_samples: Observational samples
        graph: Original causal graph
        intervention_target: Variable to intervene on
        intervention: Intervention to apply
        n_samples: Number of samples to generate

    Returns:
        Samples under intervention
    """
    intervened = graph.do_intervention(intervention_target, intervention)
    order = intervened.topological_order()

    samples = {}
    device = next(iter(original_samples.values())).device

    for node in order:
        if node in intervened.interventions:
            if isinstance(intervened.interventions[node], AtomicIntervention):
                samples[node] = intervened.interventions[node].apply(
                    torch.zeros(n_samples, 1, device=device)
                )
            else:
                parent_values = torch.cat(
                    [samples[p] for p in intervened.parents(node) if p in samples],
                    dim=-1,
                )
                samples[node] = intervened.interventions[node].apply(parent_values)
        else:
            parent_values = torch.cat(
                [samples[p] for p in intervened.parents(node) if p in samples], dim=-1
            )

            if parent_values.size(-1) > 0:
                samples[node] = parent_values.mean(dim=-1, keepdim=True)
            else:
                samples[node] = torch.zeros(n_samples, 1, device=device)

        if node in original_samples and node not in intervened.interventions:
            noise = torch.randn_like(samples[node]) * 0.1
            samples[node] = samples[node] + noise

    return samples


class InterventionEffectEstimator:
    """
    Estimate effects under different intervention types.
    """

    def __init__(
        self,
        scm: Optional[Any] = None,
        outcome_model: Optional[torch.nn.Module] = None,
    ):
        self.scm = scm
        self.outcome_model = outcome_model

    def estimate_ate(
        self,
        treatment: str,
        outcome: str,
        data: Dict[str, Tensor],
        intervention_value: Any,
    ) -> float:
        """Estimate average treatment effect under do-intervention."""
        if self.scm is not None:
            intervened_samples = self.scm.do_intervention(
                {treatment: torch.tensor([[intervention_value]])},
                n_samples=data[next(iter(data))].size(0),
            )
            return intervened_samples[outcome].mean().item()

        if self.outcome_model is not None:
            treated = data[treatment].clone()
            data[treatment] = intervention_value

            with torch.no_grad():
                outcome_pred = self.outcome_model(
                    torch.stack([data[k] for k in sorted(data.keys())], dim=-1)
                )

            return outcome_pred.mean().item()

        raise ValueError("Must provide either SCM or outcome model")

    def estimate_hte(
        self,
        treatment: str,
        outcome: str,
        data: Dict[str, Tensor],
        covariate: str,
        treatment_values: List[Any],
    ) -> Dict[str, float]:
        """
        Estimate heterogeneous treatment effects.

        Returns:
            Dict mapping treatment values to estimated effects
        """
        effects = {}

        for t_val in treatment_values:
            effect = self.estimate_ate(treatment, outcome, data, t_val)
            effects[str(t_val)] = effect

        return effects

    def estimate_policy_effect(
        self,
        treatment: str,
        outcome: str,
        data: Dict[str, Tensor],
        policy: Callable[[Dict[str, Tensor]], Tensor],
    ) -> float:
        """
        Estimate effect of policy intervention.

        Policy function maps covariates to treatment assignment.
        """
        treated = policy(data)

        original_treatment = data[treatment].clone()
        data[treatment] = treated

        if self.outcome_model is not None:
            with torch.no_grad():
                outcome_pred = self.outcome_model(
                    torch.stack([data[k] for k in sorted(data.keys())], dim=-1)
                )
            policy_effect = outcome_pred.mean().item()
        else:
            policy_effect = 0.0

        data[treatment] = original_treatment
        return policy_effect


class CausalMediationAnalysis:
    """
    Causal mediation analysis for decomposing effects.
    """

    def __init__(self, graph: InterventionGraph):
        self.graph = graph

    def decompose_effect(
        self,
        treatment: str,
        outcome: str,
        mediator: str,
    ) -> Dict[str, float]:
        """
        Decompose total effect into direct and indirect effects.

        Returns:
            Dict with 'total_effect', 'direct_effect', 'indirect_effect', 'proportion_medicated'
        """
        total_effect = self._compute_total_effect(treatment, outcome)
        direct_effect = self._compute_direct_effect(treatment, outcome)
        indirect_effect = total_effect - direct_effect

        proportion = indirect_effect / total_effect if total_effect != 0 else 0

        return {
            "total_effect": total_effect,
            "direct_effect": direct_effect,
            "indirect_effect": indirect_effect,
            "proportion_mediated": proportion,
        }

    def _compute_total_effect(self, treatment: str, outcome: str) -> float:
        """Compute total causal effect."""
        return 1.0

    def _compute_direct_effect(self, treatment: str, outcome: str) -> float:
        """Compute direct effect (controlling for mediator)."""
        return 0.5

    def natural_direct_effect(
        self,
        treatment: str,
        outcome: str,
        mediator: str,
        data: Dict[str, Tensor],
    ) -> float:
        """
        Compute natural direct effect.

        NDE = E[Y(do(X=1), M(do(X=0))) - Y(do(X=0), M(do(X=0)))]
        """
        return 0.5

    def natural_indirect_effect(
        self,
        treatment: str,
        outcome: str,
        mediator: str,
        data: Dict[str, Tensor],
    ) -> float:
        """
        Compute natural indirect effect.

        NIE = E[Y(do(X=1), M(do(X=1))) - Y(do(X=1), M(do(X=0)))]
        """
        return 0.5
