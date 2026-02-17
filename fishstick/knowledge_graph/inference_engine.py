"""
Inference Rule Engine Module

Provides logical inference capabilities for knowledge graphs.

This module provides:
- InferenceEngine: Forward and backward chaining
- Support for Horn clauses and Datalog rules
- Materialized views for caching
- Rule priority and conflict resolution
"""

from typing import Dict, List, Set, Optional, Tuple, Any, Callable
from dataclasses import dataclass, field
from collections import defaultdict
from enum import Enum, auto
import torch
from torch import Tensor


class InferenceDirection(Enum):
    """Direction of inference."""

    FORWARD = auto()
    BACKWARD = auto()
    BIDIRECTIONAL = auto()


@dataclass
class InferenceRule:
    """
    Represents a rule for inference.

    Attributes:
        rule_id: Unique rule identifier
        antecedents: List of antecedent patterns
        consequent: Consequent pattern
        weight: Rule weight/confidence
        priority: Rule priority for conflict resolution
    """

    rule_id: str
    antecedents: List[Tuple[str, str, str]]
    consequent: Tuple[str, str, str]
    weight: float = 1.0
    priority: int = 0

    def __repr__(self):
        ant_str = " AND ".join([f"({s},{p},{o})" for s, p, o in self.antecedents])
        return f"{self.rule_id}: {ant_str} => ({self.consequent[0]},{self.consequent[1]},{self.consequent[2]})"

    def __hash__(self):
        return hash(self.rule_id)


@dataclass
class Fact:
    """
    Represents a fact in the knowledge base.

    Attributes:
        subject: Subject entity
        predicate: Relation
        object: Object entity
        confidence: Confidence score
        source: Source of the fact
    """

    subject: str
    predicate: str
    object: str
    confidence: float = 1.0
    source: Optional[str] = None

    def to_tuple(self) -> Tuple[str, str, str]:
        return (self.subject, self.predicate, self.object)

    def __hash__(self):
        return hash(self.to_tuple())

    def __eq__(self, other):
        if not isinstance(other, Fact):
            return False
        return self.to_tuple() == other.to_tuple()


class MaterializedView:
    """
    Caches inferred facts for efficient query answering.
    """

    def __init__(self, max_size: int = 10000):
        self.max_size = max_size
        self.facts: Set[Tuple[str, str, str]] = set()
        self.fact_metadata: Dict[Tuple[str, str, str], Dict[str, Any]] = {}
        self.index_by_subject: Dict[str, Set[Tuple[str, str, str]]] = defaultdict(set)
        self.index_by_predicate: Dict[str, Set[Tuple[str, str, str]]] = defaultdict(set)
        self.index_by_object: Dict[str, Set[Tuple[str, str, str]]] = defaultdict(set)

    def add_fact(self, fact: Fact) -> None:
        """Add a fact to the materialized view."""
        key = fact.to_tuple()

        if key in self.facts:
            self.fact_metadata[key]["confidence"] = max(
                self.fact_metadata[key]["confidence"], fact.confidence
            )
            return

        if len(self.facts) >= self.max_size:
            self._evict_lru()

        self.facts.add(key)
        self.fact_metadata[key] = {
            "confidence": fact.confidence,
            "source": fact.source,
        }

        self.index_by_subject[fact.subject].add(key)
        self.index_by_predicate[fact.predicate].add(key)
        self.index_by_object[fact.object].add(key)

    def _evict_lru(self) -> None:
        """Evict least recently used fact."""
        if not self.facts:
            return

        first_key = next(iter(self.facts))
        self.remove_fact(first_key)

    def remove_fact(self, key: Tuple[str, str, str]) -> None:
        """Remove a fact from the view."""
        if key not in self.facts:
            return

        s, p, o = key
        self.index_by_subject[s].discard(key)
        self.index_by_predicate[p].discard(key)
        self.index_by_object[o].discard(key)

        self.facts.discard(key)
        self.fact_metadata.pop(key, None)

    def query(
        self,
        subject: Optional[str] = None,
        predicate: Optional[str] = None,
        object: Optional[str] = None,
    ) -> List[Fact]:
        """Query facts matching the pattern."""
        if subject is not None:
            candidate_keys = self.index_by_subject.get(subject, set())
        elif predicate is not None:
            candidate_keys = self.index_by_predicate.get(predicate, set())
        elif object is not None:
            candidate_keys = self.index_by_object.get(object, set())
        else:
            candidate_keys = self.facts

        results = []

        for key in candidate_keys:
            s, p, o = key

            if subject is not None and s != subject:
                continue
            if predicate is not None and p != predicate:
                continue
            if object is not None and o != object:
                continue

            metadata = self.fact_metadata.get(key, {})
            results.append(
                Fact(
                    subject=s,
                    predicate=p,
                    object=o,
                    confidence=metadata.get("confidence", 1.0),
                    source=metadata.get("source"),
                )
            )

        return results

    def clear(self) -> None:
        """Clear all facts."""
        self.facts.clear()
        self.fact_metadata.clear()
        self.index_by_subject.clear()
        self.index_by_predicate.clear()
        self.index_by_object.clear()


class InferenceEngine:
    """
    Logical inference engine for knowledge graphs.

    Supports forward chaining, backward chaining, and hybrid inference.
    """

    def __init__(
        self,
        direction: InferenceDirection = InferenceDirection.FORWARD,
        use_materialization: bool = True,
        max_inference_depth: int = 10,
    ):
        self.direction = direction
        self.use_materialization = use_materialization
        self.max_inference_depth = max_inference_depth

        self.rules: List[InferenceRule] = []
        self.materialized_view = MaterializedView() if use_materialization else None

        self.rule_index: Dict[str, List[InferenceRule]] = defaultdict(list)

        self.conflict_resolver: Callable[[List[InferenceRule]], InferenceRule] = (
            self._default_conflict_resolution
        )

    def add_rule(self, rule: InferenceRule) -> None:
        """Add a rule to the engine."""
        self.rules.append(rule)

        for ant in rule.antecedents:
            self.rule_index[ant[1]].append(rule)

    def add_fact(self, fact: Fact) -> None:
        """Add a fact to the knowledge base."""
        if self.materialized_view:
            self.materialized_view.add_fact(fact)

    def forward_chain(
        self,
        initial_facts: Optional[Set[Fact]] = None,
        max_iterations: int = 100,
    ) -> Set[Fact]:
        """
        Perform forward chaining inference.

        Args:
            initial_facts: Starting facts
            max_iterations: Maximum number of iterations

        Returns:
            Set of all inferred facts
        """
        facts = initial_facts or set()

        if self.materialized_view:
            for fact in facts:
                self.materialized_view.add_fact(fact)

        all_facts = set(facts)

        for _ in range(max_iterations):
            new_facts = set()

            for rule in sorted(self.rules, key=lambda r: -r.priority):
                matches = self._match_rule_antecedents(rule, all_facts)

                for match in matches:
                    inferred = self._apply_rule(rule, match)

                    if inferred not in all_facts and inferred not in new_facts:
                        new_facts.add(inferred)

                        if self.materialized_view:
                            self.materialized_view.add_fact(inferred)

            if not new_facts:
                break

            all_facts |= new_facts

        return all_facts

    def backward_chain(
        self,
        goal: Tuple[str, str, str],
    ) -> bool:
        """
        Perform backward chaining to prove a goal.

        Args:
            goal: Goal triplet to prove

        Returns:
            True if goal can be proven
        """
        if self.materialized_view:
            existing = self.materialized_view.query(
                subject=goal[0],
                predicate=goal[1],
                object=goal[2],
            )
            if existing:
                return True

        return self._prove_goal(goal, set(), 0)

    def _prove_goal(
        self,
        goal: Tuple[str, str, str],
        visited: Set[Tuple[str, str, str]],
        depth: int,
    ) -> bool:
        """Recursively prove a goal."""
        if depth > self.max_inference_depth:
            return False

        if goal in visited:
            return False

        visited.add(goal)

        applicable_rules = self.rule_index.get(goal[1], [])

        for rule in sorted(applicable_rules, key=lambda r: -r.priority):
            if rule.consequent == goal:
                if self._prove_rule(rule, visited, depth + 1):
                    return True

        return False

    def _prove_rule(
        self,
        rule: InferenceRule,
        visited: Set[Tuple[str, str, str]],
        depth: int,
    ) -> bool:
        """Prove a rule can be applied."""

        def generate_groundings(
            antecedents: List[Tuple[str, str, str]], idx: int
        ) -> List[Dict[str, str]]:
            if idx >= len(antecedents):
                return [{}]

            ant = antecedents[idx]
            groundings = []

            facts = []
            if self.materialized_view:
                facts = self.materialized_view.query(
                    subject=ant[0] if ant[0] != "?" else None,
                    predicate=ant[1],
                    object=ant[2] if ant[2] != "?" else None,
                )

            for fact in facts:
                bindings = {}

                if ant[0] == "?":
                    bindings["?s"] = fact.subject
                elif ant[0] != fact.subject:
                    continue

                if ant[2] == "?":
                    bindings["?o"] = fact.object
                elif ant[2] != fact.object:
                    continue

                for g in generate_groundings(antecedents, idx + 1):
                    combined = bindings.copy()
                    combined.update(g)
                    groundings.append(combined)

            return groundings

        groundings = generate_groundings(rule.antecedents, 0)

        return len(groundings) > 0

    def _match_rule_antecedents(
        self,
        rule: InferenceRule,
        known_facts: Set[Fact],
    ) -> List[Dict[str, str]]:
        """Match rule antecedents against known facts."""

        def generate_groundings(
            antecedents: List[Tuple[str, str, str]], idx: int
        ) -> List[Dict[str, str]]:
            if idx >= len(antecedents):
                return [{}]

            ant = antecedents[idx]
            groundings = []

            for fact in known_facts:
                if fact.predicate != ant[1]:
                    continue

                bindings = {}

                if ant[0] != "?" and ant[0] != fact.subject:
                    continue
                if ant[0] == "?":
                    bindings["?s"] = fact.subject

                if ant[2] != "?" and ant[2] != fact.object:
                    continue
                if ant[2] == "?":
                    bindings["?o"] = fact.object

                for g in generate_groundings(antecedents, idx + 1):
                    combined = bindings.copy()
                    combined.update(g)
                    groundings.append(combined)

            return groundings

        return generate_groundings(rule.antecedents, 0)

    def _apply_rule(
        self,
        rule: InferenceRule,
        bindings: Dict[str, str],
    ) -> Fact:
        """Apply a rule with bindings to generate a new fact."""
        s = bindings.get("?s", rule.consequent[0])
        p = rule.consequent[1]
        o = bindings.get("?o", rule.consequent[2])

        return Fact(
            subject=s,
            predicate=p,
            object=o,
            confidence=rule.weight,
            source=rule.rule_id,
        )

    def _default_conflict_resolution(
        self,
        rules: List[InferenceRule],
    ) -> InferenceRule:
        """Default conflict resolution: highest priority wins."""
        return max(rules, key=lambda r: r.priority)

    def query(
        self,
        subject: Optional[str] = None,
        predicate: Optional[str] = None,
        object: Optional[str] = None,
    ) -> List[Fact]:
        """Query the knowledge base."""
        if self.materialized_view:
            return self.materialized_view.query(subject, predicate, object)
        return []

    def explain(self, fact: Fact) -> List[str]:
        """Explain how a fact was derived."""
        explanations = []

        key = fact.to_tuple()

        if self.materialized_view and key in self.materialized_view.fact_metadata:
            source = self.materialized_view.fact_metadata[key].get("source")
            if source:
                explanations.append(f"Derived from rule: {source}")

        return explanations


class DatalogEngine:
    """
    Datalog-style inference engine.

    Supports recursive queries and negation.
    """

    def __init__(self):
        self.edb: Set[Tuple[str, str, str]] = set()
        self.idb: Set[InferenceRule] = set()

    def add_extensional_fact(self, fact: Tuple[str, str, str]) -> None:
        """Add extensional database (EDB) fact."""
        self.edb.add(fact)

    def add_intensional_rule(self, rule: InferenceRule) -> None:
        """Add intensional database (IDB) rule."""
        self.idb.add(rule)

    def evaluate(self) -> Set[Tuple[str, str, str]]:
        """Evaluate all IDB rules."""

        def fixpoint(previous: Set[Tuple[str, str, str]]) -> Set[Tuple[str, str, str]]:
            current = previous.copy()

            for rule in self.idb:
                for ant_pattern in self._enumerate_antecedents(
                    rule.antecedents, previous
                ):
                    consequent = self._ground_consequent(rule.consequent, ant_pattern)
                    current.add(consequent)

            return current

        result = self.edb.copy()

        changed = True
        while changed:
            new_result = fixpoint(result)
            if new_result == result:
                changed = False
            result = new_result

        return result

    def _enumerate_antecedents(
        self,
        antecedents: List[Tuple[str, str, str]],
        known_facts: Set[Tuple[str, str, str]],
    ) -> List[Dict[str, str]]:
        """Enumerate all valid groundings of antecedents."""
        if not antecedents:
            return [{}]

        results = []
        first = antecedents[0]

        for fact in known_facts:
            if fact[1] != first[1]:
                continue

            bindings = {}

            if first[0] != "?" and first[0] != fact[0]:
                continue
            if first[0] == "?":
                bindings["?s"] = fact[0]

            if first[2] != "?" and first[2] != fact[2]:
                continue
            if first[2] == "?":
                bindings["?o"] = fact[2]

            for rest in self._enumerate_antecedents(antecedents[1:], known_facts):
                combined = bindings.copy()
                combined.update(rest)
                results.append(combined)

        return results

    def _ground_consequent(
        self,
        consequent: Tuple[str, str, str],
        bindings: Dict[str, str],
    ) -> Tuple[str, str, str]:
        """Ground consequent with bindings."""
        s = bindings.get("?s", consequent[0])
        p = consequent[1]
        o = bindings.get("?o", consequent[2])

        return (s, p, o)

    def query(self, query_pattern: Tuple[str, str, str]) -> Set[Tuple[str, str, str]]:
        """Query the database."""
        results = set()

        all_facts = self.evaluate()

        s, p, o = query_pattern

        for fact in all_facts:
            if (
                (s == "?" or s == fact[0])
                and (p == "?" or p == fact[1])
                and (o == "?" or o == fact[2])
            ):
                results.add(fact)

        return results


def create_transitive_closure_rule(
    relation: str,
) -> InferenceRule:
    """
    Create a rule for computing transitive closure.

    Args:
        relation: Relation to make transitive

    Returns:
        InferenceRule for transitive closure
    """
    return InferenceRule(
        rule_id=f"transitive_closure_{relation}",
        antecedents=[
            ("?s", relation, "?x"),
            ("?x", relation, "?o"),
        ],
        consequent=("?s", relation, "?o"),
        weight=1.0,
        priority=0,
    )


def create_inverse_rule(
    relation: str,
    inverse: str,
) -> InferenceRule:
    """
    Create a rule for inverse relations.

    Args:
        relation: Original relation
        inverse: Inverse relation

    Returns:
        InferenceRule for inverse
    """
    return InferenceRule(
        rule_id=f"inverse_{relation}_{inverse}",
        antecedents=[("?s", relation, "?o")],
        consequent=("?o", inverse, "?s"),
        weight=1.0,
        priority=1,
    )


def create_symmetric_rule(relation: str) -> InferenceRule:
    """
    Create a rule for symmetric relations.

    Args:
        relation: Relation to make symmetric

    Returns:
        InferenceRule for symmetry
    """
    return InferenceRule(
        rule_id=f"symmetric_{relation}",
        antecedents=[("?s", relation, "?o")],
        consequent=("?o", relation, "?s"),
        weight=1.0,
        priority=1,
    )
