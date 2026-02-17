"""
Do-Calculus Operators and Causal Identification.

Implements:
- Do-calculus rules (back-door, front-door)
- Causal effect identification
- Conditional interventions
- Graph-based identification algorithms
"""

from typing import Dict, List, Optional, Set, Tuple, Callable, Any, Union
import torch
from torch import Tensor
import numpy as np
from dataclasses import dataclass
from enum import Enum
from collections import defaultdict
import itertools


class DoCalculusRules(Enum):
    """Do-calculus rules for causal identification."""

    REMOVE_IRRELEVANT = "remove_irrelevant"
    INSERT_IRRELEVANT = "insert_irrelevant"
    EXCHANGE = "exchange"
    COLLAPSING = "collapsing"


@dataclass
class Graph:
    """Represents a causal graph."""

    nodes: List[str]
    edges: List[Tuple[str, str]]
    bidirected_edges: Optional[List[Tuple[str, str]]] = None

    def parents(self, node: str) -> List[str]:
        """Get parent nodes."""
        return [src for src, dst in self.edges if dst == node]

    def children(self, node: str) -> List[str]:
        """Get child nodes."""
        return [dst for src, dst in self.edges if src == node]

    def neighbors(self, node: str) -> List[str]:
        """Get neighboring nodes (parents and children)."""
        return list(set(self.parents(node) + self.children(node)))

    def ancestors(self, node: str) -> Set[str]:
        """Get all ancestors of a node."""
        ancestors = set()
        queue = [node]

        while queue:
            current = queue.pop(0)
            for parent in self.parents(current):
                if parent not in ancestors:
                    ancestors.add(parent)
                    queue.append(parent)

        return ancestors

    def descendants(self, node: str) -> Set[str]:
        """Get all descendants of a node."""
        descendants = set()
        queue = [node]

        while queue:
            current = queue.pop(0)
            for child in self.children(current):
                if child not in descendants:
                    descendants.add(child)
                    queue.append(child)

        return descendants

    def is_path_blocked(
        self,
        path: List[str],
        conditioning_set: Set[str],
    ) -> bool:
        """Check if a path is blocked by conditioning set."""
        for i in range(1, len(path) - 1):
            middle = path[i]
            prev_node = path[i - 1]
            next_node = path[i + 1]

            is_chain = (prev_node, middle) in self.edges and (
                middle,
                next_node,
            ) in self.edges
            is_fork = (middle, prev_node) in self.edges and (
                middle,
                next_node,
            ) in self.edges
            isCollider = (prev_node, middle) not in self.edges and (
                next_node,
                middle,
            ) not in self.edges

            if is_chain or is_fork:
                if middle in conditioning_set:
                    return True
            elif isCollider:
                descendants = self.descendants(middle)
                if not descendants & conditioning_set:
                    return True

        return False

    def d_separated(
        self,
        node1: str,
        node2: str,
        conditioning_set: Set[str],
    ) -> bool:
        """Check if two nodes are d-separated given conditioning set."""
        all_paths = self._find_all_paths(node1, node2)

        for path in all_paths:
            if not self.is_path_blocked(path, conditioning_set):
                return False

        return True

    def _find_all_paths(
        self,
        start: str,
        end: str,
        path: Optional[List[str]] = None,
    ) -> List[List[str]]:
        """Find all paths between two nodes."""
        if path is None:
            path = []

        path = path + [start]

        if start == end:
            return [path]

        if start not in self.nodes:
            return []

        paths = []

        for neighbor in self.neighbors(start):
            if neighbor not in path:
                new_paths = self._find_all_paths(neighbor, end, path)
                paths.extend(new_paths)

        return paths


class DoCalculus:
    """
    Implementation of do-calculus for causal inference.

    Do-calculus provides rules for manipulating do-operators
    and identifying causal effects from observational data.
    """

    def __init__(self, graph: Graph):
        self.graph = graph

    def identify_effect(
        self,
        treatment: str,
        outcome: str,
        conditioning_set: Optional[Set[str]] = None,
    ) -> Optional[str]:
        """
        Identify causal effect P(Y | do(X)).

        Returns expression for the identified effect, or None if not identifiable.
        """
        if conditioning_set is None:
            conditioning_set = set()

        if self._back_door_criterion(treatment, outcome, conditioning_set):
            return self._back_door_adjustment(treatment, outcome, conditioning_set)

        if self._front_door_criterion(treatment, outcome):
            return self._front_door_adjustment(treatment, outcome)

        return None

    def _back_door_criterion(
        self,
        treatment: str,
        outcome: str,
        conditioning_set: Set[str],
    ) -> bool:
        """
        Check if back-door criterion is satisfied.

        A set Z satisfies back-door criterion relative to (X, Y) if:
        1. Z blocks all back-door paths from X to Y
        2. X and Z have no common descendant
        """
        back_door_paths = self._find_back_door_paths(treatment, outcome)

        for path in back_door_paths:
            if not self.graph.is_path_blocked(path, conditioning_set):
                return False

        descendants_x = self.graph.descendants(treatment)
        for z in conditioning_set:
            if z in descendants_x:
                return False

        return True

    def _front_door_criterion(
        self,
        treatment: str,
        outcome: str,
    ) -> bool:
        """
        Check if front-door criterion is satisfied.

        A variable Z satisfies front-door criterion if:
        1. Z intercepts all directed paths from X to Y
        2. X and Y are d-separated given Z
        3. No back-door path from X to Z is open given Z
        """
        directed_paths = self._find_directed_paths(treatment, outcome)

        if not directed_paths:
            return False

        all_z = set()
        for path in directed_paths:
            all_z.update(path[1:-1])

        if not all_z:
            return False

        if not self.graph.d_separated(treatment, outcome, all_z):
            return False

        for z in all_z:
            back_door_paths = self._find_back_door_paths(treatment, z)
            if back_door_paths:
                return False

        return True

    def _back_door_adjustment(
        self,
        treatment: str,
        outcome: str,
        conditioning_set: Set[str],
    ) -> str:
        """Generate back-door adjustment formula."""
        if not conditioning_set:
            return f"P({outcome} | do({treatment}))"

        z_str = ", ".join(sorted(conditioning_set))
        return f"sum_z P({outcome} | {treatment}, z) * P(z)"

    def _front_door_adjustment(
        self,
        treatment: str,
        outcome: str,
    ) -> str:
        """Generate front-door adjustment formula."""
        return f"sum_z P(z | {treatment}) * sum_t P({outcome} | t, z) * P(t)"

    def _find_back_door_paths(
        self,
        treatment: str,
        outcome: str,
    ) -> List[List[str]]:
        """Find all back-door paths (paths starting with arrow into treatment)."""
        paths = []

        for neighbor in self.graph.parents(treatment):
            if neighbor != outcome:
                all_paths = self.graph._find_all_paths(neighbor, outcome)
                for path in all_paths:
                    if (
                        path[0] == treatment
                        or (neighbor, treatment) in self.graph.edges
                    ):
                        paths.append([treatment] + path)

        return paths

    def _find_directed_paths(
        self,
        start: str,
        end: str,
    ) -> List[List[str]]:
        """Find all directed paths from start to end."""
        paths = []

        def dfs(current: str, path: List[str]):
            if current == end:
                paths.append(path.copy())
                return

            for child in self.graph.children(current):
                if child not in path:
                    path.append(child)
                    dfs(child, path)
                    path.pop()

        dfs(start, [start])
        return paths

    def apply_rule(
        self,
        expression: str,
        rule: DoCalculusRules,
        **kwargs,
    ) -> str:
        """Apply a do-calculus rule to an expression."""
        if rule == DoCalculusRules.REMOVE_IRRELEVANT:
            return self._remove_irrelevant(expression, kwargs.get("target"))
        elif rule == DoCalculusRules.INSERT_IRRELEVANT:
            return self._insert_irrelevant(expression, kwargs.get("target"))
        elif rule == DoCalculusRules.EXCHANGE:
            return self._exchange(expression, kwargs.get("var"))
        elif rule == DoCalculusRules.COLLAPSING:
            return self._collapsing(expression)

        return expression

    def _remove_irrelevant(self, expression: str, target: str) -> str:
        """Remove irrelevant variable from conditioning."""
        return expression

    def _insert_irrelevant(self, expression: str, target: str) -> str:
        """Insert irrelevant variable into conditioning."""
        return expression

    def _exchange(self, expression: str, var: str) -> str:
        """Exchange do-operator with conditioning."""
        return expression

    def _collapsing(self, expression: str) -> str:
        """Apply collapsing rule."""
        return expression


class BackDoorCriterion:
    """
    Back-door criterion checker and effect identifier.
    """

    def __init__(self, graph: Graph):
        self.graph = graph

    def is_satisfied(
        self,
        treatment: str,
        outcome: str,
        conditioning_set: Set[str],
    ) -> bool:
        """Check if back-door criterion is satisfied."""
        do_calc = DoCalculus(self.graph)
        return do_calc._back_door_criterion(treatment, outcome, conditioning_set)

    def find_valid_adjustment_sets(
        self,
        treatment: str,
        outcome: str,
        available_set: Optional[Set[str]] = None,
    ) -> List[Set[str]]:
        """Find all valid adjustment sets."""
        if available_set is None:
            available_set = set(self.graph.nodes) - {treatment, outcome}

        valid_sets = []

        for r in range(len(available_set) + 1):
            for subset in itertools.combinations(available_set, r):
                conditioning_set = set(subset)
                if self.is_satisfied(treatment, outcome, conditioning_set):
                    valid_sets.append(conditioning_set)

        return valid_sets

    def minimal_adjustment_set(
        self,
        treatment: str,
        outcome: str,
    ) -> Optional[Set[str]]:
        """Find minimal adjustment set."""
        valid_sets = self.find_valid_adjustment_sets(treatment, outcome)

        if not valid_sets:
            return None

        min_set = min(valid_sets, key=len)
        return min_set


class FrontDoorCriterion:
    """
    Front-door criterion checker and effect identifier.
    """

    def __init__(self, graph: Graph):
        self.graph = graph

    def is_satisfied(
        self,
        treatment: str,
        outcome: str,
    ) -> bool:
        """Check if front-door criterion is satisfied."""
        do_calc = DoCalculus(self.graph)
        return do_calc._front_door_criterion(treatment, outcome)

    def identify_effect(
        self,
        treatment: str,
        outcome: str,
    ) -> Optional[str]:
        """Identify effect using front-door adjustment."""
        if not self.is_satisfied(treatment, outcome):
            return None

        do_calc = DoCalculus(self.graph)
        return do_calc._front_door_adjustment(treatment, outcome)


@dataclass
class ConditionalIntervention:
    """
    Conditional intervention: do(X = x) given Z = z.
    """

    treatment: str
    value: Any
    condition: Optional[Tuple[str, Any]] = None

    def __str__(self) -> str:
        if self.condition:
            return f"do({self.treatment} = {self.value} | {self.condition[0]} = {self.condition[1]})"
        return f"do({self.treatment} = {self.value})"


def identify_causal_effect(
    graph: Graph,
    treatment: str,
    outcome: str,
    method: str = "auto",
) -> Optional[str]:
    """
    Identify causal effect using available methods.

    Args:
        graph: Causal graph
        treatment: Treatment variable
        outcome: Outcome variable
        method: Identification method ('backdoor', 'frontdoor', 'auto')

    Returns:
        Identification formula or None if not identifiable
    """
    if method == "auto":
        backdoor = BackDoorCriterion(graph)
        if backdoor.is_satisfied(treatment, outcome, set()):
            return f"P({outcome} | do({treatment})) = sum_z P({outcome} | {treatment}, z) * P(z)"

        frontdoor = FrontDoorCriterion(graph)
        if frontdoor.is_satisfied(treatment, outcome):
            return frontdoor.identify_effect(treatment, outcome)

        return None

    elif method == "backdoor":
        backdoor = BackDoorCriterion(graph)
        minimal = backdoor.minimal_adjustment_set(treatment, outcome)
        if minimal:
            z_str = ", ".join(sorted(minimal))
            return f"P({outcome} | do({treatment})) = sum_{z_str} P({outcome} | {treatment}, z) * P(z)"
        return None

    elif method == "frontdoor":
        frontdoor = FrontDoorCriterion(graph)
        return frontdoor.identify_effect(treatment, outcome)

    return None


def is_adjustment_set_valid(
    graph: Graph,
    treatment: str,
    outcome: str,
    adjustment_set: Set[str],
) -> bool:
    """
    Check if an adjustment set is valid for identifying causal effect.
    """
    backdoor = BackDoorCriterion(graph)
    return backdoor.is_satisfied(treatment, outcome, adjustment_set)


class CausalEffectIdentifier:
    """
    Complete causal effect identification using multiple methods.
    """

    def __init__(self, graph: Graph):
        self.graph = graph
        self.backdoor = BackDoorCriterion(graph)
        self.frontdoor = FrontDoorCriterion(graph)

    def identify(
        self,
        treatment: str,
        outcome: str,
    ) -> Dict[str, Any]:
        """
        Attempt to identify causal effect using all methods.

        Returns:
            Dict with identification results
        """
        results = {
            "identifiable": False,
            "methods_tried": [],
            "formula": None,
            "adjustment_set": None,
        }

        if treatment not in self.graph.nodes or outcome not in self.graph.nodes:
            return results

        minimal = self.backdoor.minimal_adjustment_set(treatment, outcome)
        if minimal:
            results["identifiable"] = True
            results["methods_tried"].append("backdoor")
            results["adjustment_set"] = minimal
            z_str = ", ".join(sorted(minimal))
            results["formula"] = f"sum_{z_str} P({outcome} | {treatment}, z) * P(z)"
            return results

        if self.frontdoor.is_satisfied(treatment, outcome):
            results["identifiable"] = True
            results["methods_tried"].append("frontdoor")
            results["formula"] = self.frontdoor.identify_effect(treatment, outcome)
            return results

        return results

    def all_valid_adjustment_sets(
        self,
        treatment: str,
        outcome: str,
    ) -> List[Set[str]]:
        """Get all valid adjustment sets."""
        return self.backdoor.find_valid_adjustment_sets(treatment, outcome)
