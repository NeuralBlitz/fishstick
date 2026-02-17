"""
Causal Dependency Graph Utilities.

Provides:
- Causal DAG representations and operations
- D-separation queries
- Ancestor/descendant queries
- Markov blanket computation
- Minimal adjustment set identification
"""

from typing import Dict, List, Optional, Set, Tuple, Callable, Any, Union
import torch
from torch import Tensor
import numpy as np
from dataclasses import dataclass, field
from collections import defaultdict, deque
from abc import ABC, abstractmethod
import itertools


@dataclass
class CausalDAG:
    """
    Directed Acyclic Graph for causal modeling.

    Provides efficient operations for:
    - Ancestor/descendant queries
    - D-separation checks
    - Markov blanket computation
    - Adjustment set identification
    """

    n_nodes: int
    adjacency: np.ndarray
    node_names: Optional[List[str]] = None
    node_types: Optional[Dict[str, str]] = field(default_factory=dict)

    def __post_init__(self):
        if self.node_names is None:
            self.node_names = [f"X{i}" for i in range(self.n_nodes)]

        self.name_to_idx = {name: i for i, name in enumerate(self.node_names)}

        if not self._is_dag():
            raise ValueError("Graph must be a DAG")

    def _is_dag(self) -> bool:
        """Check if graph is acyclic."""
        try:
            self.topological_sort()
            return True
        except ValueError:
            return False

    def parents(self, node: Union[int, str]) -> List[int]:
        """Get parent indices of a node."""
        if isinstance(node, str):
            node = self.name_to_idx[node]
        return list(np.where(self.adjacency[:, node] == 1)[0])

    def children(self, node: Union[int, str]) -> List[int]:
        """Get child indices of a node."""
        if isinstance(node, str):
            node = self.name_to_idx[node]
        return list(np.where(self.adjacency[node, :] == 1)[0])

    def neighbors(self, node: Union[int, str]) -> List[int]:
        """Get neighboring nodes (parents and children)."""
        return list(set(self.parents(node) + self.children(node)))

    def ancestors(
        self,
        node: Union[int, str],
        include_self: bool = False,
    ) -> Set[int]:
        """Get all ancestors of a node."""
        if isinstance(node, str):
            node = self.name_to_idx[node]

        ancestors = set()
        queue = [node]

        while queue:
            current = queue.pop(0)

            for parent in self.parents(current):
                if parent not in ancestors:
                    ancestors.add(parent)
                    queue.append(parent)

        if include_self:
            ancestors.add(node)

        return ancestors

    def descendants(
        self,
        node: Union[int, str],
        include_self: bool = False,
    ) -> Set[int]:
        """Get all descendants of a node."""
        if isinstance(node, str):
            node = self.name_to_idx[node]

        descendants = set()
        queue = [node]

        while queue:
            current = queue.pop(0)

            for child in self.children(current):
                if child not in descendants:
                    descendants.add(child)
                    queue.append(child)

        if include_self:
            descendants.add(node)

        return descendants

    def topological_sort(self) -> List[int]:
        """Get topological ordering of nodes."""
        in_degree = np.sum(self.adjacency, axis=0).tolist()
        queue = deque([i for i in range(self.n_nodes) if in_degree[i] == 0])
        order = []

        while queue:
            node = queue.popleft()
            order.append(node)

            for child in self.children(node):
                in_degree[child] -= 1
                if in_degree[child] == 0:
                    queue.append(child)

        if len(order) != self.n_nodes:
            raise ValueError("Graph contains a cycle")

        return order

    def all_paths(
        self,
        source: Union[int, str],
        target: Union[int, str],
    ) -> List[List[int]]:
        """Find all directed paths from source to target."""
        if isinstance(source, str):
            source = self.name_to_idx[source]
        if isinstance(target, str):
            target = self.name_to_idx[target]

        paths = []

        def dfs(current: int, path: List[int]):
            if current == target:
                paths.append(path.copy())
                return

            for child in self.children(current):
                if child not in path:
                    path.append(child)
                    dfs(child, path)
                    path.pop()

        dfs(source, [source])
        return paths

    def all_backdoor_paths(
        self,
        treatment: Union[int, str],
        outcome: Union[int, str],
    ) -> List[List[int]]:
        """Find all back-door paths from treatment to outcome."""
        if isinstance(treatment, str):
            treatment = self.name_to_idx[treatment]
        if isinstance(outcome, str):
            outcome = self.name_to_idx[outcome]

        paths = []

        for parent in self.parents(treatment):

            def dfs(current: int, path: List[int], blocked_by_treatment: bool):
                if current == outcome and path:
                    paths.append(path.copy())
                    return

                for neighbor in self.neighbors(current):
                    if neighbor in path:
                        continue

                    if neighbor == treatment:
                        continue

                    if blocked_by_treatment and (current, neighbor) in zip(
                        path, [path[-1]] * len([neighbor])
                    ):
                        continue

                    edge_toward_treatment = (neighbor, treatment) in [
                        (self.node_names[i], self.node_names[j])
                        for i, j in self._edge_list()
                    ]

                    new_blocked = blocked_by_treatment or edge_toward_treatment

                    path.append(neighbor)
                    dfs(neighbor, path, new_blocked)
                    path.pop()

            dfs(parent, [treatment, parent], False)

        return paths

    def _edge_list(self) -> List[Tuple[int, int]]:
        """Get list of edges as (source, target) pairs."""
        edges = []
        for i in range(self.n_nodes):
            for j in range(self.n_nodes):
                if self.adjacency[i, j] == 1:
                    edges.append((i, j))
        return edges

    def is_blocked(
        self,
        path: List[int],
        conditioning_set: Set[int],
    ) -> bool:
        """
        Check if a path is blocked by conditioning set.

        A path is blocked if:
        1. A non-collider on the path is in conditioning set, OR
        2. A collider on the path is NOT in conditioning set and has no descendants in conditioning set
        """
        for i in range(1, len(path) - 1):
            node = path[i]
            prev_node = path[i - 1]
            next_node = path[i + 1]

            is_collider = (
                self.adjacency[prev_node, node] == 0
                and self.adjacency[next_node, node] == 0
                and self.adjacency[node, prev_node] == 0
                and self.adjacency[node, next_node] == 0
            )

            is_chain_or_fork = not is_collider

            if is_chain_or_fork:
                if node in conditioning_set:
                    return True
            else:
                descendants = self.descendants(node)
                if not descendants & conditioning_set:
                    return True

        return False

    def d_separated(
        self,
        node1: Union[int, str],
        node2: Union[int, str],
        conditioning_set: Set[Union[int, str]],
    ) -> bool:
        """
        Check if two nodes are d-separated given conditioning set.

        Two nodes are d-separated if all paths between them are blocked.
        """
        if isinstance(node1, str):
            node1 = self.name_to_idx[node1]
        if isinstance(node2, str):
            node2 = self.name_to_idx[node2]

        conditioning_set_idx = set()
        for node in conditioning_set:
            if isinstance(node, str):
                conditioning_set_idx.add(self.name_to_idx[node])
            else:
                conditioning_set_idx.add(node)

        all_paths = self._find_all_paths(node1, node2)

        for path in all_paths:
            if not self.is_blocked(path, conditioning_set_idx):
                return False

        return True

    def _find_all_paths(
        self,
        start: int,
        end: int,
        path: Optional[List[int]] = None,
    ) -> List[List[int]]:
        """Find all paths between two nodes (not just directed)."""
        if path is None:
            path = [start]

        if start == end and len(path) > 1:
            return [path]

        paths = []

        for neighbor in self.neighbors(start):
            if neighbor not in path:
                new_paths = self._find_all_paths(neighbor, end, path + [neighbor])
                paths.extend(new_paths)

        return paths


def d_separation(
    graph: CausalDAG,
    node1: Union[int, str],
    node2: Union[int, str],
    conditioning_set: Set[Union[int, str]],
) -> bool:
    """
    Check d-separation between two nodes.

    Args:
        graph: Causal DAG
        node1: First node
        node2: Second node
        conditioning_set: Set of conditioned nodes

    Returns:
        True if d-separated
    """
    return graph.d_separated(node1, node2, conditioning_set)


class AncestorQuery:
    """Efficient ancestor/descendant queries."""

    def __init__(self, dag: CausalDAG):
        self.dag = dag
        self._ancestor_cache = {}
        self._descendant_cache = {}
        self._build_caches()

    def _build_caches(self) -> None:
        """Pre-compute ancestor/descendant sets."""
        for node in range(self.dag.n_nodes):
            self._ancestor_cache[node] = self.dag.ancestors(node, include_self=False)
            self._descendant_cache[node] = self.dag.descendants(
                node, include_self=False
            )

    def is_ancestor(
        self,
        potential_ancestor: Union[int, str],
        node: Union[int, str],
    ) -> bool:
        """Check if potential_ancestor is an ancestor of node."""
        if isinstance(potential_ancestor, str):
            potential_ancestor = self.dag.name_to_idx[potential_ancestor]
        if isinstance(node, str):
            node = self.dag.name_to_idx[node]

        return potential_ancestor in self._ancestor_cache[node]

    def is_descendant(
        self,
        potential_descendant: Union[int, str],
        node: Union[int, str],
    ) -> bool:
        """Check if potential_descendant is a descendant of node."""
        if isinstance(potential_descendant, str):
            potential_descendant = self.dag.name_to_idx[potential_descendant]
        if isinstance(node, str):
            node = self.dag.name_to_idx[node]

        return potential_descendant in self._descendant_cache[node]

    def get_ancestors(
        self,
        node: Union[int, str],
        include_self: bool = False,
    ) -> Set[int]:
        """Get all ancestors of a node."""
        if isinstance(node, str):
            node = self.dag.name_to_idx[node]

        anc = self._ancestor_cache[node].copy()
        if include_self:
            anc.add(node)
        return anc

    def get_descendants(
        self,
        node: Union[int, str],
        include_self: bool = False,
    ) -> Set[int]:
        """Get all descendants of a node."""
        if isinstance(node, str):
            node = self.dag.name_to_idx[node]

        desc = self._descendant_cache[node].copy()
        if include_self:
            desc.add(node)
        return desc


def is_valid_adjustment_set(
    graph: CausalDAG,
    treatment: Union[int, str],
    outcome: Union[int, str],
    adjustment_set: Set[Union[int, str]],
) -> bool:
    """
    Check if adjustment set is valid for identifying causal effect.

    A set Z is a valid adjustment set if it blocks all back-door
    paths from treatment to outcome and contains no descendants of treatment.
    """
    if isinstance(treatment, str):
        treatment = graph.name_to_idx[treatment]
    if isinstance(outcome, str):
        outcome = graph.name_to_idx[outcome]

    adjustment_idx = set()
    for node in adjustment_set:
        if isinstance(node, str):
            adjustment_idx.add(graph.name_to_idx[node])
        else:
            adjustment_idx.add(node)

    treatment_descendants = graph.descendants(treatment)
    if adjustment_idx & treatment_descendants:
        return False

    backdoor_paths = graph.all_backdoor_paths(treatment, outcome)

    for path in backdoor_paths:
        if not graph.is_blocked(path, adjustment_idx):
            return False

    return True


def minimal_adjustment_set(
    graph: CausalDAG,
    treatment: Union[int, str],
    outcome: Union[int, str],
) -> Optional[Set[int]]:
    """
    Find minimal set of variables that blocks all back-door paths.
    """
    if isinstance(treatment, str):
        treatment = graph.name_to_idx[treatment]
    if isinstance(outcome, str):
        outcome = graph.name_to_idx[outcome]

    confounders = set()
    for parent in graph.parents(treatment):
        confounders.add(parent)

    for ancestor in graph.ancestors(outcome):
        if ancestor != treatment:
            if graph.d_separated(treatment, outcome, {ancestor}):
                continue
            confounders.add(ancestor)

    return confounders if confounders else None


def markov_blanket(
    graph: CausalDAG,
    node: Union[int, str],
) -> Set[int]:
    """
    Compute Markov blanket of a node.

    The Markov blanket consists of:
    - Parents
    - Children
    - Spouses (parents of children)
    """
    if isinstance(node, str):
        node = graph.name_to_idx[node]

    blanket = set()

    blanket.update(graph.parents(node))
    blanket.update(graph.children(node))

    for child in graph.children(node):
        blanket.update(graph.parents(child))

    blanket.discard(node)

    return blanket


def causal_ordering(graph: CausalDAG) -> List[int]:
    """Get causal ordering (topological sort)."""
    return graph.topological_sort()


class CausalGraphAnalyzer:
    """
    Comprehensive analysis of causal graph properties.
    """

    def __init__(self, dag: CausalDAG):
        self.dag = dag
        self.ancestor_query = AncestorQuery(dag)

    def identify_effect(
        self,
        treatment: Union[int, str],
        outcome: Union[int, str],
    ) -> Dict[str, Any]:
        """Identify causal effect if possible."""
        if isinstance(treatment, str):
            treatment = self.dag.name_to_idx[treatment]
        if isinstance(outcome, str):
            outcome = self.dag.name_to_idx[outcome]

        result = {
            "identifiable": False,
            "method": None,
            "adjustment_set": None,
        }

        minimal = minimal_adjustment_set(self.dag, treatment, outcome)
        if minimal is not None:
            result["identifiable"] = True
            result["method"] = "backdoor"
            result["adjustment_set"] = minimal
            return result

        return result

    def get_frontdoor_variables(
        self,
        treatment: Union[int, str],
        outcome: Union[int, str],
    ) -> List[int]:
        """Find variables that satisfy front-door criterion."""
        if isinstance(treatment, str):
            treatment = self.dag.name_to_idx[treatment]
        if isinstance(outcome, str):
            outcome = self.dag.name_to_idx[outcome]

        candidates = []

        for node in range(self.dag.n_nodes):
            if node == treatment or node == outcome:
                continue

            directed_paths = self.dag.all_paths(treatment, node)

            if not directed_paths:
                continue

            blocks_outcome = self.dag.d_separated(treatment, outcome, {node})

            if not blocks_outcome:
                candidates.append(node)

        return candidates

    def get_instrumental_variables(
        self,
        treatment: Union[int, str],
        outcome: Union[int, str],
    ) -> List[int]:
        """Find instrumental variables."""
        if isinstance(treatment, str):
            treatment = self.dag.name_to_idx[treatment]
        if isinstance(outcome, str):
            outcome = self.dag.name_to_idx[outcome]

        instruments = []

        for node in range(self.dag.n_nodes):
            if node == treatment or node == outcome:
                continue

            has_direct_path = len(self.dag.all_paths(node, treatment)) > 0

            not_adjacent_treatment = node not in self.dag.neighbors(treatment)

            d_sep_outcome = self.dag.d_separated(node, outcome, {treatment})

            if has_direct_path and not_adjacent_treatment and d_sep_outcome:
                instruments.append(node)

        return instruments

    def confounding_paths(
        self,
        treatment: Union[int, str],
        outcome: Union[int, str],
    ) -> List[Dict[str, Any]]:
        """Analyze all back-door paths (potential confounding)."""
        if isinstance(treatment, str):
            treatment = self.dag.name_to_idx[treatment]
        if isinstance(outcome, str):
            outcome = self.dag.name_to_idx[outcome]

        backdoor_paths = self.dag.all_backdoor_paths(treatment, outcome)

        paths_info = []

        for path in backdoor_paths:
            path_info = {
                "path": path,
                "nodes": [self.dag.node_names[i] for i in path],
                "colliders": [],
            }

            for i in range(1, len(path) - 1):
                node = path[i]
                prev_node = path[i - 1]
                next_node = path[i + 1]

                is_collider = (
                    self.dag.adjacency[prev_node, node] == 0
                    and self.dag.adjacency[next_node, node] == 0
                )

                if is_collider:
                    path_info["colliders"].append(node)

            paths_info.append(path_info)

        return paths_info
