import torch
import torch.nn as nn
import numpy as np
from typing import Dict, List, Optional, Tuple, Set, Any
from dataclasses import dataclass, field
from enum import Enum


class EdgeType(Enum):
    DIRECTED = "->"
    BIDIRECTED = "<->"
    UNDIRECTED = "-"


@dataclass
class CausalGraph:
    nodes: List[str] = field(default_factory=list)
    edges: List[Tuple[str, str]] = field(default_factory=list)
    edge_type: Dict[Tuple[str, str], EdgeType] = field(default_factory=dict)
    node_types: Dict[str, str] = field(default_factory=dict)
    node_data: Dict[str, torch.Tensor] = field(default_factory=dict)

    def __post_init__(self):
        if self.nodes and not self.node_types:
            self.node_types = {node: "observed" for node in self.nodes}

    def add_node(self, node: str, node_type: str = "observed") -> None:
        if node not in self.nodes:
            self.nodes.append(node)
            self.node_types[node] = node_type

    def add_edge(
        self, source: str, target: str, edge_type: EdgeType = EdgeType.DIRECTED
    ) -> None:
        self.add_node(source)
        self.add_node(target)
        self.edges.append((source, target))
        self.edge_type[(source, target)] = edge_type

    def get_parents(self, node: str) -> List[str]:
        return [src for src, tgt in self.edges if tgt == node]

    def get_children(self, node: str) -> List[str]:
        return [tgt for src, tgt in self.edges if src == node]

    def get_neighbors(self, node: str) -> List[str]:
        parents = self.get_parents(node)
        children = self.get_children(node)
        return list(set(parents + children))

    def get_descendants(self, node: str) -> Set[str]:
        descendants = set()
        children = self.get_children(node)
        for child in children:
            descendants.add(child)
            descendants.update(self.get_descendants(child))
        return descendants

    def get_ancestors(self, node: str) -> Set[str]:
        ancestors = set()
        parents = self.get_parents(node)
        for parent in parents:
            ancestors.add(parent)
            ancestors.update(self.get_ancestors(parent))
        return ancestors

    def to_adjacency_matrix(self) -> torch.Tensor:
        n = len(self.nodes)
        node_idx = {node: i for i, node in enumerate(self.nodes)}
        adj = torch.zeros(n, n)
        for src, tgt in self.edges:
            if src in node_idx and tgt in node_idx:
                adj[node_idx[src], node_idx[tgt]] = 1
        return adj

    def to_dag(self) -> bool:
        adj = self.to_adjacency_matrix()
        n = adj.shape[0]
        visited = torch.zeros(n, dtype=torch.bool)
        rec_stack = torch.zeros(n, dtype=torch.bool)

        def has_cycle(node: int) -> bool:
            visited[node] = True
            rec_stack[node] = True
            neighbors = torch.where(adj[node] == 1)[0]
            for neighbor in neighbors:
                if not visited[neighbor]:
                    if has_cycle(neighbor):
                        return True
                elif rec_stack[neighbor]:
                    return True
            rec_stack[node] = False
            return False

        for i in range(n):
            if not visited[i]:
                if has_cycle(i):
                    return False
        return True


@dataclass
class CausalIdentification:
    graph: CausalGraph
    treatment: str = ""
    outcome: str = ""
    confounders: List[str] = field(default_factory=list)

    def is_identifiable(
        self, treatment: str, outcome: str, backdoor_criterion: bool = True
    ) -> bool:
        if backdoor_criterion:
            return self._check_backdoor(treatment, outcome)
        return self._check_frontdoor(treatment, outcome)

    def _check_backdoor(self, treatment: str, outcome: str) -> bool:
        ancestors_t = self.graph.get_ancestors(treatment)
        ancestors_o = self.graph.get_ancestors(outcome)
        common = ancestors_t.intersection(ancestors_o)

        for conf in common:
            path = self._find_path(conf, treatment, exclude=[outcome])
            if path and self._is_blocked(path, treatment, outcome):
                return False
        return True

    def _check_frontdoor(self, treatment: str, outcome: str) -> bool:
        children_t = self.graph.get_children(treatment)
        if not children_t:
            return False
        for mediator in children_t:
            if self.graph.get_parents(mediator) == [treatment]:
                ancestors_o = self.graph.get_ancestors(outcome)
                if treatment not in ancestors_o:
                    return True
        return False

    def _find_path(
        self, start: str, end: str, exclude: List[str] = []
    ) -> Optional[List[str]]:
        visited = set()
        queue = [(start, [start])]

        while queue:
            node, path = queue.pop(0)
            if node == end:
                return path
            if node in visited or node in exclude:
                continue
            visited.add(node)

            for neighbor in self.graph.get_neighbors(node):
                if neighbor not in visited and neighbor not in exclude:
                    queue.append((neighbor, path + [neighbor]))
        return None

    def _is_blocked(self, path: List[str], treatment: str, outcome: str) -> bool:
        for i in range(1, len(path) - 1):
            node = path[i]
            parents = self.graph.get_parents(node)
            if node == treatment or node == outcome:
                return False
            if len(parents) <= 1:
                return True
        return False

    def get_adjustment_set(self, treatment: str, outcome: str) -> Set[str]:
        if self._check_backdoor(treatment, outcome):
            ancestors_t = self.graph.get_ancestors(treatment)
            ancestors_o = self.graph.get_ancestors(outcome)
            return ancestors_t.intersection(ancestors_o)
        return set()

    def get_instrumental_variables(self, treatment: str, outcome: str) -> List[str]:
        instruments = []
        for node in self.graph.nodes:
            if node == treatment or node == outcome:
                continue
            children_t = self.graph.get_children(treatment)
            ancestors_t = self.graph.get_ancestors(treatment)

            if node in ancestors_t:
                continue
            if any(node in self.graph.get_ancestors(c) for c in children_t):
                continue
            if self._check_backdoor(node, outcome):
                instruments.append(node)
        return instruments


class EstimatorMethod(Enum):
    LINEAR_REGRESSION = "linear_regression"
    DIFFERENCE_IN_DIFFERENCES = "did"
    INVERSE_PROBABILITY_WEIGHTING = "ipw"
    PROPENSITY_SCORE_MATCHING = "psm"


@dataclass
class CausalEstimation:
    graph: CausalGraph
    treatment: str
    outcome: str
    data: torch.Tensor
    node_to_idx: Dict[str, int]
    method: EstimatorMethod = EstimatorMethod.LINEAR_REGRESSION

    def estimate(self, adjustment_set: Optional[Set[str]] = None) -> Dict[str, float]:
        if adjustment_set is None:
            adjustment_set = set()

        if self.method == EstimatorMethod.LINEAR_REGRESSION:
            return self._linear_regression(adjustment_set)
        elif self.method == EstimatorMethod.INVERSE_PROBABILITY_WEIGHTING:
            return self._ipw(adjustment_set)
        elif self.method == EstimatorMethod.PROPENSITY_SCORE_MATCHING:
            return self._psm(adjustment_set)
        return {"causal_effect": 0.0, "std_error": 0.0}

    def _linear_regression(self, adjustment_set: Set[str]) -> Dict[str, float]:
        treat_idx = self.node_to_idx[self.treatment]
        outcome_idx = self.node_to_idx[self.outcome]

        y = self.data[:, outcome_idx]
        t = self.data[:, treat_idx]

        features = [t]
        for node in adjustment_set:
            if node in self.node_to_idx:
                features.append(self.data[:, self.node_to_idx[node]])

        X = torch.stack(features, dim=1)
        X = torch.cat([torch.ones(X.shape[0], 1), X], dim=1)

        try:
            beta = torch.linalg.lstsq(X, y).solution
            effect = beta[1].item()
            residuals = y - X @ beta
            n = X.shape[0]
            k = X.shape[1]
            mse = (residuals**2).sum() / (n - k)
            var = mse * torch.linalg.inv(X.T @ X).diag()
            std_error = torch.sqrt(var[1]).item()
        except:
            effect = 0.0
            std_error = 0.0

        return {"causal_effect": effect, "std_error": std_error}

    def _ipw(self, adjustment_set: Set[str]) -> Dict[str, float]:
        treat_idx = self.node_to_idx[self.treatment]
        outcome_idx = self.node_to_idx[self.outcome]

        y = self.data[:, outcome_idx]
        t = self.data[:, treat_idx]

        confounders = []
        for node in adjustment_set:
            if node in self.node_to_idx:
                confounders.append(self.data[:, self.node_to_idx[node]])

        if confounders:
            X = torch.stack(confounders, dim=1)
            X = torch.cat([torch.ones(X.shape[0], 1), X], dim=1)

            prop_model = nn.Linear(X.shape[1], 1)
            optimizer = torch.optim.Adam(prop_model.parameters(), lr=0.01)

            for _ in range(200):
                logits = prop_model(X).squeeze()
                loss = nn.functional.binary_cross_entropy_with_logits(
                    logits, t, reduction="mean"
                )
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            propensity = torch.sigmoid(prop_model(X).squeeze())
            propensity = torch.clamp(propensity, 0.01, 0.99)

            weights = t / propensity + (1 - t) / (1 - propensity)
            effect = (
                (y * weights).mean() - (y * (1 - weights / weights.mean())).mean()
            ).item()
        else:
            effect = (y[t == 1].mean() - y[t == 0].mean()).item()

        return {"causal_effect": effect, "std_error": 0.0}

    def _psm(self, adjustment_set: Set[str]) -> Dict[str, float]:
        treat_idx = self.node_to_idx[self.treatment]
        outcome_idx = self.node_to_idx[self.outcome]

        y = self.data[:, outcome_idx]
        t = self.data[:, treat_idx]

        confounders = []
        for node in adjustment_set:
            if node in self.node_to_idx:
                confounders.append(self.data[:, self.node_to_idx[node]])

        if not confounders:
            return {"causal_effect": 0.0, "std_error": 0.0}

        X = torch.stack(confounders, dim=1)

        treated_mask = t == 1
        control_mask = t == 0

        treated_X = X[treated_mask]
        control_X = X[control_mask]
        treated_y = y[treated_mask]
        control_y = y[control_mask]

        dist_matrix = torch.cdist(treated_X, control_X)
        min_idx = dist_matrix.argmin(dim=1)

        matched_control_y = control_y[min_idx]
        effect = (treated_y - matched_control_y).mean().item()

        return {"causal_effect": effect, "std_error": 0.0}


class RefutationMethod(Enum):
    PLACEBO_TEST = "placebo"
    RANDOM_COMMON_CAUSE = "random_common_cause"
    SUBSAMPLE = "subsample"
    BOOTSTRAP = "bootstrap"


@dataclass
class CausalRefutation:
    estimation: CausalEstimation
    original_effect: float
    method: RefutationMethod = RefutationMethod.BOOTSTRAP
    n_simulations: int = 100

    def refute(self) -> Dict[str, Any]:
        if self.method == RefutationMethod.PLACEBO_TEST:
            return self._placebo_test()
        elif self.method == RefutationMethod.RANDOM_COMMON_CAUSE:
            return self._random_common_cause()
        elif self.method == RefutationMethod.SUBSAMPLE:
            return self._subsample()
        elif self.method == RefutationMethod.BOOTSTRAP:
            return self._bootstrap()
        return {"refuted": False, "p_value": 1.0}

    def _placebo_test(self) -> Dict[str, Any]:
        effects = []
        for _ in range(self.n_simulations):
            shuffled_data = self.estimation.data.clone()
            treat_col = self.estimation.node_to_idx[self.estimation.treatment]
            shuffled_data[:, treat_col] = shuffled_data[:, treat_col][
                torch.randperm(shuffled_data.shape[0])
            ]

            temp_est = CausalEstimation(
                self.estimation.graph,
                self.estimation.treatment,
                self.estimation.outcome,
                shuffled_data,
                self.estimation.node_to_idx,
                self.estimation.method,
            )
            result = temp_est.estimate()
            effects.append(result["causal_effect"])

        effects = torch.tensor(effects)
        p_value = (
            (torch.abs(effects) >= torch.abs(torch.tensor(self.original_effect)))
            .float()
            .mean()
        )

        return {
            "refuted": p_value < 0.05,
            "p_value": p_value.item(),
            "effect_distribution": effects.mean().item(),
        }

    def _random_common_cause(self) -> Dict[str, Any]:
        effects = []
        for _ in range(self.n_simulations):
            augmented_data = self.estimation.data.clone()
            random_noise = torch.randn(augmented_data.shape[0])
            augmented_data = torch.cat(
                [augmented_data, random_noise.unsqueeze(1)], dim=1
            )

            temp_est = CausalEstimation(
                self.estimation.graph,
                self.estimation.treatment,
                self.estimation.outcome,
                augmented_data,
                self.estimation.node_to_idx,
                self.estimation.method,
            )
            result = temp_est.estimate()
            effects.append(result["causal_effect"])

        effects = torch.tensor(effects)
        p_value = (torch.abs(effects - self.original_effect) > 0.1).float().mean()

        return {"refuted": p_value < 0.05, "p_value": p_value.item()}

    def _subsample(self) -> Dict[str, Any]:
        effects = []
        n_samples = self.estimation.data.shape[0]

        for _ in range(self.n_simulations):
            subsample_size = int(n_samples * 0.7)
            indices = torch.randperm(n_samples)[:subsample_size]
            subsampled_data = self.estimation.data[indices]

            temp_est = CausalEstimation(
                self.estimation.graph,
                self.estimation.treatment,
                self.estimation.outcome,
                subsampled_data,
                self.estimation.node_to_idx,
                self.estimation.method,
            )
            result = temp_est.estimate()
            effects.append(result["causal_effect"])

        effects = torch.tensor(effects)
        effect_variance = effects.var().item()

        return {
            "refuted": effect_variance > 0.1,
            "variance": effect_variance,
            "ci_lower": (effects.mean() - 1.96 * effects.std()).item(),
            "ci_upper": (effects.mean() + 1.96 * effects.std()).item(),
        }

    def _bootstrap(self) -> Dict[str, Any]:
        effects = []
        n_samples = self.estimation.data.shape[0]

        for _ in range(self.n_simulations):
            indices = torch.randint(0, n_samples, (n_samples,))
            bootstrapped_data = self.estimation.data[indices]

            temp_est = CausalEstimation(
                self.estimation.graph,
                self.estimation.treatment,
                self.estimation.outcome,
                bootstrapped_data,
                self.estimation.node_to_idx,
                self.estimation.method,
            )
            result = temp_est.estimate()
            effects.append(result["causal_effect"])

        effects = torch.tensor(effects)
        se = effects.std().item()
        z = self.original_effect / se if se > 0 else 0
        p_value = 2 * (1 - torch.distributions.Normal(0, 1).cdf(torch.tensor(abs(z))))

        return {
            "refuted": p_value < 0.05,
            "p_value": p_value.item()
            if not torch.is_tensor(p_value)
            else p_value.item(),
            "std_error": se,
            "ci_lower": (self.original_effect - 1.96 * se),
            "ci_upper": (self.original_effect + 1.96 * se),
        }
