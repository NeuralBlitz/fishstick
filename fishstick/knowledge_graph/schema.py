"""
Graph Schema Validation Module

Provides schema validation and inference for knowledge graphs.

This module provides:
- SchemaValidator: Validate graphs against schema
- Entity type validation
- Relation domain/range validation
- Cardinality constraint checking
- Schema inference from existing graphs
- Cycle detection
"""

from typing import Dict, List, Set, Optional, Tuple, Any, Callable
from dataclasses import dataclass, field
from enum import Enum, auto
from collections import defaultdict
import torch
from torch import Tensor


class ConstraintType(Enum):
    """Types of schema constraints."""

    DOMAIN = auto()
    RANGE = auto()
    CARDINALITY = auto()
    INVERSE = auto()
    SYMMETRIC = auto()
    TRANSITIVE = auto()
    REFLEXIVE = auto()
    TYPE = auto()
    UNIQUE = auto()


@dataclass
class SchemaConstraint:
    """
    Represents a schema constraint.

    Attributes:
        constraint_type: Type of constraint
        relation: Relation name (if applicable)
        params: Constraint parameters
        description: Human-readable description
    """

    constraint_type: ConstraintType
    relation: Optional[str] = None
    params: Dict[str, Any] = field(default_factory=dict)
    description: str = ""


@dataclass
class Schema:
    """
    Knowledge graph schema definition.

    Attributes:
        entity_types: Set of valid entity types
        relation_signatures: Dict mapping relation to (domain_type, range_type)
        cardinality_constraints: Dict of (relation, (min, max)) cardinalities
        inverse_relations: Dict mapping relations to their inverses
        symmetric_relations: Set of symmetric relations
        transitive_relations: Set of transitive relations
    """

    entity_types: Set[str] = field(default_factory=set)
    relation_signatures: Dict[str, Tuple[str, str]] = field(default_factory=dict)
    cardinality_constraints: Dict[str, Tuple[int, int]] = field(default_factory=dict)
    inverse_relations: Dict[str, str] = field(default_factory=dict)
    symmetric_relations: Set[str] = field(default_factory=set)
    transitive_relations: Set[str] = field(default_factory=set)
    reflexive_entities: Dict[str, Set[str]] = field(default_factory=dict)


@dataclass
class ValidationResult:
    """
    Result of schema validation.

    Attributes:
        is_valid: Whether validation passed
        errors: List of validation errors
        warnings: List of validation warnings
        statistics: Additional statistics
    """

    is_valid: bool
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    statistics: Dict[str, Any] = field(default_factory=dict)


class SchemaValidator:
    """
    Validates knowledge graphs against schema constraints.
    """

    def __init__(self, schema: Optional[Schema] = None):
        self.schema = schema or Schema()
        self.error_handlers: Dict[ConstraintType, Callable] = {}

    def set_schema(self, schema: Schema) -> None:
        """Set the schema for validation."""
        self.schema = schema

    def validate_entity_types(
        self,
        entities: Dict[str, Tuple[str, str]],
    ) -> ValidationResult:
        """
        Validate entity types.

        Args:
            entities: Dict mapping entity_id to (entity_name, entity_type)

        Returns:
            ValidationResult
        """
        errors = []
        warnings = []

        unknown_types = set()

        for entity_id, (name, entity_type) in entities.items():
            if entity_type not in self.schema.entity_types:
                unknown_types.add(entity_type)

        if unknown_types:
            errors.append(f"Unknown entity types: {unknown_types}")

        return ValidationResult(
            is_valid=len(errors) == 0,
            errors=errors,
            warnings=warnings,
        )

    def validate_relation_signatures(
        self,
        relations: Dict[Tuple[str, str, str], str],
    ) -> ValidationResult:
        """
        Validate relation domain and range.

        Args:
            relations: Dict mapping (source, target, relation) to relation_type

        Returns:
            ValidationResult
        """
        errors = []
        warnings = []

        signature_violations = []

        for (source, target, rel_name), rel_type in relations.items():
            if rel_name not in self.schema.relation_signatures:
                continue

            expected_domain, expected_range = self.schema.relation_signatures[rel_name]

            if rel_type != expected_domain:
                signature_violations.append(
                    f"Relation {rel_name}: source has type {rel_type}, expected {expected_domain}"
                )

            if rel_type != expected_range:
                signature_violations.append(
                    f"Relation {rel_name}: target has type {rel_type}, expected {expected_range}"
                )

        errors.extend(signature_violations)

        return ValidationResult(
            is_valid=len(errors) == 0,
            errors=errors,
            warnings=warnings,
        )

    def validate_cardinality(
        self,
        relations: Dict[str, List[str]],
    ) -> ValidationResult:
        """
        Validate cardinality constraints.

        Args:
            relations: Dict mapping relation to list of target entities

        Returns:
            ValidationResult
        """
        errors = []
        warnings = []

        for rel_name, targets in relations.items():
            if rel_name not in self.schema.cardinality_constraints:
                continue

            min_card, max_card = self.schema.cardinality_constraints[rel_name]
            num_targets = len(targets)

            if num_targets < min_card:
                errors.append(
                    f"Relation {rel_name}: has {num_targets} targets, minimum is {min_card}"
                )

            if max_card > 0 and num_targets > max_card:
                errors.append(
                    f"Relation {rel_name}: has {num_targets} targets, maximum is {max_card}"
                )

        return ValidationResult(
            is_valid=len(errors) == 0,
            errors=errors,
            warnings=warnings,
        )

    def validate_inverse_relations(
        self,
        triplets: List[Tuple[str, str, str]],
    ) -> ValidationResult:
        """
        Validate inverse relation consistency.

        Args:
            triplets: List of (subject, predicate, object) triplets

        Returns:
            ValidationResult
        """
        errors = []
        warnings = []

        relation_pairs = defaultdict(set)

        for s, p, o in triplets:
            relation_pairs[p].add((s, o))

        for rel, inverse_rel in self.schema.inverse_relations.items():
            if inverse_rel not in relation_pairs:
                warnings.append(f"Inverse relation {inverse_rel} not found for {rel}")
                continue

            for s, o in relation_pairs[rel]:
                if (o, s) not in relation_pairs[inverse_rel]:
                    warnings.append(
                        f"Inverse relation missing: ({s}, {rel}, {o}) but not ({o}, {inverse_rel}, {s})"
                    )

        return ValidationResult(
            is_valid=len(errors) == 0,
            errors=errors,
            warnings=warnings,
        )

    def validate_symmetric_relations(
        self,
        triplets: List[Tuple[str, str, str]],
    ) -> ValidationResult:
        """
        Validate symmetric relation consistency.

        Args:
            triplets: List of (subject, predicate, object) triplets

        Returns:
            ValidationResult
        """
        errors = []
        warnings = []

        relation_pairs = defaultdict(set)

        for s, p, o in triplets:
            relation_pairs[p].add((s, o))

        for rel in self.schema.symmetric_relations:
            for s, o in relation_pairs[rel]:
                if (o, s) not in relation_pairs[rel]:
                    warnings.append(
                        f"Symmetric relation {rel} violated: ({s}, {rel}, {o}) but not ({o}, {rel}, {s})"
                    )

        return ValidationResult(
            is_valid=len(errors) == 0,
            errors=errors,
            warnings=warnings,
        )

    def validate_transitive_relations(
        self,
        triplets: List[Tuple[str, str, str]],
    ) -> ValidationResult:
        """
        Validate transitive relation consistency.

        Args:
            triplets: List of (subject, predicate, object) triplets

        Returns:
            ValidationResult
        """
        errors = []
        warnings = []

        relation_graph: Dict[str, Set[str]] = defaultdict(set)

        for s, p, o in triplets:
            if p in self.schema.transitive_relations:
                relation_graph[s].add(o)

        for rel in self.schema.transitive_relations:
            implied = self._compute_transitive_closure(relation_graph, rel, triplets)

            for s, o in relation_graph[rel]:
                if o not in implied.get(s, set()):
                    warnings.append(
                        f"Transitive relation {rel} incomplete: ({s}, {rel}, {o}) implied"
                    )

        return ValidationResult(
            is_valid=len(errors) == 0,
            errors=errors,
            warnings=warnings,
        )

    def _compute_transitive_closure(
        self,
        graph: Dict[str, Set[str]],
        rel: str,
        triplets: List[Tuple[str, str, str]],
    ) -> Dict[str, Set[str]]:
        """Compute transitive closure for a relation."""
        closure: Dict[str, Set[str]] = defaultdict(set)

        for s, p, o in triplets:
            if p == rel:
                closure[s].add(o)

        changed = True
        while changed:
            changed = False
            for s in list(closure.keys()):
                for mid in list(closure[s]):
                    if mid in closure:
                        new_targets = closure[mid] - closure[s]
                        if new_targets:
                            closure[s] |= new_targets
                            changed = True

        return closure

    def validate_all(
        self,
        entities: Dict[str, Tuple[str, str]],
        triplets: List[Tuple[str, str, str]],
    ) -> ValidationResult:
        """
        Run all validations.

        Args:
            entities: Dict of entity_id -> (name, type)
            triplets: List of triplets

        Returns:
            Combined ValidationResult
        """
        all_errors = []
        all_warnings = []
        all_stats = {}

        entity_result = self.validate_entity_types(entities)
        all_errors.extend(entity_result.errors)
        all_warnings.extend(entity_result.warnings)

        relation_dict = {}
        for s, p, o in triplets:
            relation_dict[(s, o, p)] = s

        rel_result = self.validate_relation_signatures(relation_dict)
        all_errors.extend(rel_result.errors)
        all_warnings.extend(rel_result.warnings)

        relations_by_name: Dict[str, List[str]] = defaultdict(list)
        for s, p, o in triplets:
            relations_by_name[p].append(o)

        card_result = self.validate_cardinality(relations_by_name)
        all_errors.extend(card_result.errors)
        all_warnings.extend(card_result.warnings)

        inverse_result = self.validate_inverse_relations(triplets)
        all_warnings.extend(inverse_result.warnings)

        symmetric_result = self.validate_symmetric_relations(triplets)
        all_warnings.extend(symmetric_result.warnings)

        transitive_result = self.validate_transitive_relations(triplets)
        all_warnings.extend(transitive_result.warnings)

        all_stats = {
            "num_entities": len(entities),
            "num_triplets": len(triplets),
            "num_errors": len(all_errors),
            "num_warnings": len(all_warnings),
        }

        return ValidationResult(
            is_valid=len(all_errors) == 0,
            errors=all_errors,
            warnings=all_warnings,
            statistics=all_stats,
        )


class SchemaInference:
    """
    Infers schema from existing knowledge graphs.
    """

    def __init__(self):
        pass

    def infer_schema(
        self,
        entities: Dict[str, str],
        triplets: List[Tuple[str, str, str]],
        min_frequency: int = 2,
    ) -> Schema:
        """
        Infer schema from graph data.

        Args:
            entities: Dict mapping entity_id to entity_type
            triplets: List of triplets
            min_frequency: Minimum frequency for inference

        Returns:
            Inferred Schema
        """
        schema = Schema()

        schema.entity_types = set(entities.values())

        relation_signatures = self._infer_relation_signatures(entities, triplets)
        schema.relation_signatures = relation_signatures

        schema.symmetric_relations = self._infer_symmetric_relations(
            triplets, min_frequency
        )

        schema.transitive_relations = self._infer_transitive_relations(
            triplets, min_frequency
        )

        schema.inverse_relations = self._infer_inverse_relations(
            triplets, min_frequency
        )

        return schema

    def _infer_relation_signatures(
        self,
        entities: Dict[str, str],
        triplets: List[Tuple[str, str, str]],
    ) -> Dict[str, Tuple[str, str]]:
        """Infer relation domain and range types."""
        relation_domains: Dict[str, Set[str]] = defaultdict(set)
        relation_ranges: Dict[str, Set[str]] = defaultdict(set)

        for s, p, o in triplets:
            if s in entities:
                relation_domains[p].add(entities[s])
            if o in entities:
                relation_ranges[p].add(entities[o])

        signatures = {}

        for rel in relation_domains:
            domains = relation_domains[rel]
            ranges = relation_ranges[rel]

            if domains and ranges:
                domain_type = list(domains)[0] if len(domains) == 1 else "entity"
                range_type = list(ranges)[0] if len(ranges) == 1 else "entity"

                signatures[rel] = (domain_type, range_type)

        return signatures

    def _infer_symmetric_relations(
        self,
        triplets: List[Tuple[str, str, str]],
        min_frequency: int,
    ) -> Set[str]:
        """Infer symmetric relations."""
        relation_pairs: Dict[str, Set[Tuple[str, str]]] = defaultdict(set)

        for s, p, o in triplets:
            relation_pairs[p].add((s, o))

        symmetric = set()

        for rel, pairs in relation_pairs.items():
            symmetric_count = 0
            for s, o in pairs:
                if (o, s) in pairs:
                    symmetric_count += 1

            if symmetric_count >= min_frequency:
                symmetric.add(rel)

        return symmetric

    def _infer_transitive_relations(
        self,
        triplets: List[Tuple[str, str, str]],
        min_frequency: int,
    ) -> Set[str]:
        """Infer transitive relations."""
        transitive_candidates: Dict[str, Dict[str, Set[str]]] = defaultdict(
            lambda: defaultdict(set)
        )

        for s, p, o in triplets:
            transitive_candidates[p][s].add(o)

        transitive = set()

        for rel, targets in transitive_candidates.items():
            for s in targets:
                for mid in targets[s]:
                    if mid in targets:
                        if targets[mid]:
                            transitive.add(rel)
                            break

        return transitive

    def _infer_inverse_relations(
        self,
        triplets: List[Tuple[str, str, str]],
        min_frequency: int,
    ) -> Dict[str, str]:
        """Infer inverse relations."""
        relation_pairs: Dict[str, Set[Tuple[str, str]]] = defaultdict(set)

        for s, p, o in triplets:
            relation_pairs[p].add((s, o))

        potential_inverses: Dict[str, Dict[str, int]] = defaultdict(
            lambda: defaultdict(int)
        )

        for rel1, pairs1 in relation_pairs.items():
            for rel2, pairs2 in relation_pairs.items():
                if rel1 != rel2:
                    inverse_count = sum(1 for s, o in pairs1 if (o, s) in pairs2)

                    if inverse_count >= min_frequency:
                        potential_inverses[rel1][rel2] = inverse_count

        inverses = {}

        for rel, candidates in potential_inverses.items():
            if candidates:
                best_inverse = max(candidates.keys(), key=lambda x: candidates[x])
                inverses[rel] = best_inverse

        return inverses


class CycleDetector:
    """
    Detects cycles in knowledge graph relations.
    """

    def __init__(self):
        pass

    def find_cycles(
        self,
        triplets: List[Tuple[str, str, str]],
        max_cycle_length: int = 5,
    ) -> List[List[Tuple[str, str, str]]]:
        """
        Find cycles in the knowledge graph.

        Args:
            triplets: List of triplets
            max_cycle_length: Maximum cycle length to find

        Returns:
            List of cycles (each cycle is list of triplets)
        """
        graph: Dict[str, List[Tuple[str, str]]] = defaultdict(list)

        for s, p, o in triplets:
            graph[s].append((o, p))

        cycles = []

        def dfs(
            current: str,
            start: str,
            path: List[Tuple[str, str, str]],
            visited: Set[str],
        ):
            if len(path) > max_cycle_length:
                return

            if current == start and len(path) > 2:
                cycles.append(path.copy())
                return

            for neighbor, rel in graph[current]:
                if neighbor == start and len(path) > 2:
                    cycles.append(path + [(current, rel, neighbor)])
                elif neighbor not in visited:
                    visited.add(neighbor)
                    dfs(neighbor, start, path + [(current, rel, neighbor)], visited)
                    visited.remove(neighbor)

        for node in graph:
            dfs(node, node, [], {node})

        return cycles

    def has_cycle(
        self,
        triplets: List[Tuple[str, str, str]],
    ) -> bool:
        """
        Check if graph has any cycle.

        Args:
            triplets: List of triplets

        Returns:
            True if cycle exists
        """
        graph: Dict[str, List[str]] = defaultdict(list)

        for s, p, o in triplets:
            graph[s].append(o)

        visited = set()
        rec_stack = set()

        def has_cycle_util(node: str) -> bool:
            visited.add(node)
            rec_stack.add(node)

            for neighbor in graph[node]:
                if neighbor not in visited:
                    if has_cycle_util(neighbor):
                        return True
                elif neighbor in rec_stack:
                    return True

            rec_stack.remove(node)
            return False

        for node in graph:
            if node not in visited:
                if has_cycle_util(node):
                    return True

        return False
