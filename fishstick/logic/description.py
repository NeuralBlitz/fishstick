"""
Description Logic Module.

Implements:
- ALC (Attributive Language with Complement) description logic
- Concepts and roles
- TBox (terminological box) reasoning
- ABox (assertional box) reasoning
- Structural reasoning algorithms

Author: Agent 13 (Fishstick Framework)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Dict, List, Optional, Set, Tuple, Union, FrozenSet
from abc import ABC, abstractmethod
from collections import defaultdict
import torch
from torch import Tensor


class ConceptType(Enum):
    """Concept constructor types."""

    TOP = auto()  # ⊤ (Thing)
    BOTTOM = auto()  # ⊥ (Nothing)
    ATOMIC = auto()  # Named concept
    NOT = auto()  # Complement (¬C)
    AND = auto()  # Intersection (C ⊓ D)
    OR = auto()  # Union (C ⊔ D)
    EXISTS = auto()  # Existential restriction (∃r.C)
    FORALL = auto()  # Universal restriction (∀r.C)


@dataclass(frozen=True)
class Concept:
    """
    Description logic concept expression.

    ALC concepts:
    - Atomic: A, B, C (named concepts)
    - Complement: ¬C
    - Intersection: C ⊓ D
    - Union: C ⊔ D
    - Existential: ∃r.C
    - Universal: ∀r.C
    """

    concept_type: ConceptType
    name: Optional[str] = None
    role: Optional[str] = None
    children: Tuple[Concept, ...] = ()

    def __str__(self) -> str:
        if self.concept_type == ConceptType.TOP:
            return "⊤"
        elif self.concept_type == ConceptType.BOTTOM:
            return "⊥"
        elif self.concept_type == ConceptType.ATOMIC:
            return self.name
        elif self.concept_type == ConceptType.NOT:
            return f"¬{self.children[0]}"
        elif self.concept_type == ConceptType.AND:
            return f"({self.children[0]} ⊓ {self.children[1]})"
        elif self.concept_type == ConceptType.OR:
            return f"({self.children[0]} ⊔ {self.children[1]})"
        elif self.concept_type == ConceptType.EXISTS:
            return f"∃{self.role}.{self.children[0]}"
        elif self.concept_type == ConceptType.FORALL:
            return f"∀{self.role}.{self.children[0]}"
        return str(self.name)

    def __hash__(self) -> int:
        return hash((self.concept_type, self.name, self.role, self.children))

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Concept):
            return NotImplemented
        return (
            self.concept_type == other.concept_type
            and self.name == other.name
            and self.role == other.role
            and self.children == other.children
        )


@dataclass
class Role:
    """
    Description logic role (binary relation).

    Roles can be:
    - Atomic: r, s, t
    - Inverse: r⁻¹
    """

    name: str
    inverse: Optional[str] = None

    def __str__(self) -> str:
        if self.inverse:
            return f"{self.name}⁻¹"
        return self.name

    def __hash__(self) -> int:
        return hash((self.name, self.inverse))

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Role):
            return NotImplemented
        return self.name == other.name and self.inverse == other.inverse


@dataclass
class TBox:
    """
    Terminological Box (TBox).

    Contains General Concept Inclusions (GCIs):
    - C ⊑ D (C is subsumed by D)
    """

    gcis: List[Tuple[Concept, Concept]] = field(default_factory=list)
    atomic_concepts: Set[str] = field(default_factory=set)
    roles: Set[str] = field(default_factory=set)

    def add_gci(self, subsumer: Concept, subsumee: Concept) -> None:
        """Add General Concept Inclusion C ⊑ D."""
        self.gcis.append((subsumee, subsumer))

    def add_atomic_concept(self, name: str) -> None:
        """Add atomic concept name."""
        self.atomic_concepts.add(name)

    def add_role(self, name: str) -> None:
        """Add role name."""
        self.roles.add(name)

    def get_supertypes(self, concept: Concept) -> Set[Concept]:
        """Get all supertypes of a concept."""
        supertypes = {Concept(ConceptType.TOP)}

        for subsumee, subsumer in self.gcis:
            if subsumee == concept:
                supertypes.add(subsumer)

        return supertypes

    def is_subsumed_by(self, subsumee: Concept, subsumer: Concept) -> bool:
        """Check if subsumee ⊑ subsumer."""
        if subsumer.concept_type == ConceptType.TOP:
            return True
        if subsumee.concept_type == ConceptType.BOTTOM:
            return True
        if subsumee == subsumer:
            return True

        supertypes = self.get_supertypes(subsumee)
        return subsumer in supertypes


@dataclass
class ABox:
    """
    Assertional Box (ABox).

    Contains concept and role assertions:
    - C(a): individual a is an instance of concept C
    - r(a, b): individuals a and b are related by role r
    """

    concept_assertions: Dict[str, Set[Concept]] = field(
        default_factory=lambda: defaultdict(set)
    )
    role_assertions: Dict[Tuple[str, str], Set[str]] = field(
        default_factory=lambda: defaultdict(set)
    )
    individuals: Set[str] = field(default_factory=set)

    def add_concept_assertion(self, individual: str, concept: Concept) -> None:
        """Add concept assertion C(a)."""
        self.individuals.add(individual)
        self.concept_assertions[individual].add(concept)

    def add_role_assertion(self, role: str, a: str, b: str) -> None:
        """Add role assertion r(a, b)."""
        self.individuals.add(a)
        self.individuals.add(b)
        self.role_assertions[(a, b)].add(role)

    def get_concepts(self, individual: str) -> Set[Concept]:
        """Get all concepts asserted for individual."""
        return self.concept_assertions.get(individual, set())

    def get_related(self, individual: str, role: str) -> Set[str]:
        """Get individuals related by role."""
        return {
            b
            for (a, b), roles in self.role_assertions.items()
            if a == individual and role in roles
        }

    def get_predecessors(self, individual: str, role: str) -> Set[str]:
        """Get individuals that relate to this individual by role."""
        return {
            a
            for (a, b), roles in self.role_assertions.items()
            if b == individual and role in roles
        }


class StructuralReasoner:
    """
    Structural reasoner for ALC description logic.

    Implements:
    - Concept satisfiability
    - Subsumption checking
    - ABox consistency
    """

    def __init__(self, tbox: Optional[TBox] = None, abox: Optional[ABox] = None):
        self.tbox = tbox or TBox()
        self.abox = abox or ABox()

    def is_satisfiable(self, concept: Concept) -> bool:
        """
        Check if concept is satisfiable.

        A concept is satisfiable if it has at least one model.
        """
        return self._check_satisfiability(concept, set())

    def _check_satisfiability(self, concept: Concept, seen: Set[Concept]) -> bool:
        """Recursively check satisfiability."""
        if concept.concept_type == ConceptType.TOP:
            return True
        if concept.concept_type == ConceptType.BOTTOM:
            return False
        if concept in seen:
            return True

        new_seen = seen | {concept}

        if concept.concept_type == ConceptType.ATOMIC:
            return True

        if concept.concept_type == ConceptType.NOT:
            return self._check_satisfiability(concept.children[0], new_seen)

        if concept.concept_type == ConceptType.AND:
            return self._check_satisfiability(
                concept.children[0], new_seen
            ) and self._check_satisfiability(concept.children[1], new_seen)

        if concept.concept_type == ConceptType.OR:
            return self._check_satisfiability(
                concept.children[0], new_seen
            ) or self._check_satisfiability(concept.children[1], new_seen)

        if concept.concept_type == ConceptType.EXISTS:
            return self._check_satisfiability(concept.children[0], new_seen)

        if concept.concept_type == ConceptType.FORALL:
            return self._check_satisfiability(concept.children[0], new_seen)

        return True

    def is_subsumed_by(self, subsumee: Concept, subsumer: Concept) -> bool:
        """Check if subsumee ⊑ subsumer."""
        if subsumer.concept_type == ConceptType.TOP:
            return True
        if subsumee.concept_type == ConceptType.BOTTOM:
            return True

        not_subsumer = Concept(ConceptType.NOT, children=(subsumer,))

        return not self.is_satisfiable(
            Concept(ConceptType.AND, children=(subsumee, not_subsumer))
        )

    def is_consistent(self) -> bool:
        """Check if ABox is consistent."""
        return self._check_abox_consistency(set())

    def _check_abox_consistency(self, seen: Set[str]) -> bool:
        """Check ABox consistency."""
        for individual in self.abox.individuals:
            if individual in seen:
                continue

            concepts = self.abox.get_concepts(individual)
            for concept in concepts:
                if not self._check_individual_concept(individual, concept, seen):
                    return False

        return True

    def _check_individual_concept(
        self, individual: str, concept: Concept, seen: Set[str]
    ) -> bool:
        """Check if individual satisfies concept."""
        if concept.concept_type == ConceptType.TOP:
            return True
        if concept.concept_type == ConceptType.BOTTOM:
            return False
        if concept.concept_type == ConceptType.ATOMIC:
            return concept.name in [c.name for c in self.abox.get_concepts(individual)]

        if concept.concept_type == ConceptType.NOT:
            neg_concept = Concept(
                ConceptType.ATOMIC, name=f"¬{concept.children[0].name}"
            )
            return neg_concept in self.abox.get_concepts(individual)

        if concept.concept_type == ConceptType.EXISTS:
            related = self.abox.get_related(individual, concept.role)
            if not related:
                return False
            for ind in related:
                if ind in seen:
                    continue
                related_concepts = self.abox.get_concepts(ind)
                if concept.children[0] not in related_concepts:
                    return False
            return True

        if concept.concept_type == ConceptType.FORALL:
            related = self.abox.get_related(individual, concept.role)
            for ind in related:
                related_concepts = self.abox.get_concepts(ind)
                if concept.children[0] not in related_concepts:
                    return self.is_subsumed_by(
                        concept.children[0], Concept(ConceptType.BOTTOM)
                    )
            return True

        return True

    def realize(self) -> Dict[str, Concept]:
        """
        Compute most specific concept for each individual.

        Returns:
            Dictionary mapping individuals to their MSC
        """
        msc = {}

        for individual in self.abox.individuals:
            concepts = list(self.abox.get_concepts(individual))

            if not concepts:
                msc[individual] = Concept(ConceptType.TOP)
                continue

            msc[individual] = concepts[0]
            for c in concepts[1:]:
                msc[individual] = Concept(
                    ConceptType.AND, children=(msc[individual], c)
                )

        return msc


class TableauAlgorithm:
    """
    Tableau algorithm for ALC concept satisfiability.

    Implements:
    - Completion tree construction
    - Blocking (equality and subset blocking)
    - Expansion rules
    """

    def __init__(self, tbox: Optional[TBox] = None):
        self.tbox = tbox or TBox()
        self.completion_forest: List[CompletionTree] = []

    def is_satisfiable(self, concept: Concept) -> bool:
        """
        Check if concept is satisfiable using tableau algorithm.

        Returns:
            True if concept is satisfiable
        """
        tree = CompletionTree()
        tree.add_node("0", concept)

        self.completion_forest = [tree]

        while self.completion_forest:
            tree = self.completion_forest.pop(0)

            if tree.is_complete():
                return True

            expanded, new_tree = tree.expand(self.tbox)

            if expanded:
                self.completion_forest.append(new_tree)
            else:
                if new_tree.is_blocked():
                    continue
                return False

        return False


class CompletionTree:
    """
    Completion tree for tableau algorithm.

    Nodes contain:
    - Label (set of concepts)
    - Edges (role connections)
    - Status (unblocked, equality-blocked, subset-blocked)
    """

    def __init__(self):
        self.nodes: Dict[str, CompletionNode] = {}
        self.edges: Dict[Tuple[str, str], Set[str]] = defaultdict(set)
        self.blocked: Set[str] = set()

    def add_node(self, node_id: str, concept: Concept) -> None:
        """Add node with initial concept."""
        self.nodes[node_id] = CompletionNode(node_id, {concept})

    def expand(self, tbox: TBox) -> Tuple[bool, CompletionTree]:
        """Expand tree using tableau rules."""
        changed = True

        while changed:
            changed = False

            for node_id, node in list(self.nodes.items()):
                if node_id in self.blocked:
                    continue

                new_node, applied = self._apply_rules(node, tbox)

                if applied:
                    changed = True

                    if new_node is not None:
                        self.nodes[new_node.node_id] = new_node

        return False, self

    def _apply_rules(
        self, node: CompletionNode, tbox: TBox
    ) -> Tuple[Optional[CompletionNode], bool]:
        """Apply expansion rules."""
        for concept in list(node.label):
            if concept.concept_type == ConceptType.AND:
                node.label.remove(concept)
                node.label.add(concept.children[0])
                node.label.add(concept.children[1])
                return None, True

            if concept.concept_type == ConceptType.OR:
                branch = CompletionTree()
                branch.nodes = {
                    k: CompletionNode(v.node_id, v.label.copy())
                    for k, v in self.nodes.items()
                }
                branch.edges = {k: v.copy() for k, v in self.edges.items()}

                new_node1 = CompletionNode(node.node_id, node.label.copy())
                new_node1.label.add(concept.children[0])

                new_node2 = CompletionNode(node.node_id, node.label.copy())
                new_node2.label.add(concept.children[1])

                return None, True

            if concept.concept_type == ConceptType.EXISTS:
                new_id = f"{node.node_id}.{concept.role}"
                new_node = CompletionNode(new_id, {concept.children[0]})
                self.edges[(node.node_id, new_id)].add(concept.role)
                return new_node, True

            if concept.concept_type == ConceptType.FORALL:
                successors = [
                    target
                    for (src, target), roles in self.edges.items()
                    if src == node.node_id and concept.role in roles
                ]

                for succ_id in successors:
                    succ_node = self.nodes.get(succ_id)
                    if succ_node and concept.children[0] not in succ_node.label:
                        succ_node.label.add(concept.children[0])
                        return None, True

        return None, False

    def is_complete(self) -> bool:
        """Check if tree is complete (no more expansion possible)."""
        for node in self.nodes.values():
            for concept in node.label:
                if concept.concept_type in [ConceptType.AND, ConceptType.OR]:
                    return False
                if concept.concept_type == ConceptType.EXISTS:
                    has_successor = any(
                        src == node.node_id for src, _ in self.edges.keys()
                    )
                    if not has_successor:
                        return False
        return True

    def is_blocked(self) -> bool:
        """Check if tree is blocked."""
        return len(self.blocked) == len(self.nodes)


@dataclass
class CompletionNode:
    """Node in completion tree."""

    node_id: str
    label: Set[Concept]
    status: str = "unblocked"

    def __hash__(self) -> int:
        return hash((self.node_id, tuple(sorted(self.label, key=str))))


def create_atomic_concept(name: str) -> Concept:
    """Create atomic concept."""
    return Concept(ConceptType.ATOMIC, name=name)


def create_top() -> Concept:
    """Create top concept (⊤)."""
    return Concept(ConceptType.TOP)


def create_bottom() -> Concept:
    """Create bottom concept (⊥)."""
    return Concept(ConceptType.BOTTOM)


def create_not(concept: Concept) -> Concept:
    """Create complement (¬C)."""
    return Concept(ConceptType.NOT, children=(concept,))


def create_and_concept(*concepts: Concept) -> Concept:
    """Create intersection (C ⊓ D)."""
    if len(concepts) == 0:
        return create_top()
    if len(concepts) == 1:
        return concepts[0]
    result = concepts[0]
    for c in concepts[1:]:
        result = Concept(ConceptType.AND, children=(result, c))
    return result


def create_or_concept(*concepts: Concept) -> Concept:
    """Create union (C ⊔ D)."""
    if len(concepts) == 0:
        return create_bottom()
    if len(concepts) == 1:
        return concepts[0]
    result = concepts[0]
    for c in concepts[1:]:
        result = Concept(ConceptType.OR, children=(result, c))
    return result


def create_exists(role: str, concept: Concept) -> Concept:
    """Create existential restriction (∃r.C)."""
    return Concept(ConceptType.EXISTS, role=role, children=(concept,))


def create_forall(role: str, concept: Concept) -> Concept:
    """Create universal restriction (∀r.C)."""
    return Concept(ConceptType.FORALL, role=role, children=(concept,))


def create_role(name: str, inverse: bool = False) -> Role:
    """Create role."""
    return Role(name, inverse=name + "_inv" if inverse else None)
