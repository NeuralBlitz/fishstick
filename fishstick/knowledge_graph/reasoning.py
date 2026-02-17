"""
Relation Reasoning Module

Implements logical and path-based reasoning over knowledge graphs.

This module provides:
- RuleMining: Discover Horn rules from graph patterns
- PathRanking: Path-based relation reasoning
- LogicalInference: Forward/backward chaining engine
- QueryEmbedding: Complex query answering
"""

from typing import Dict, List, Set, Tuple, Optional, Callable, Any
from dataclasses import dataclass, field
from collections import defaultdict
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from itertools import combinations


@dataclass
class HornRule:
    """
    Represents a Horn rule in the form: B1 AND B2 AND ... => H

    Attributes:
        head: Head predicate (conclusion)
        body: List of body predicates (antecedents)
        confidence: Rule confidence score
        support: Number of supporting triplets
        coverage: Fraction of head covered by rule
    """

    head: Tuple[str, str, str]
    body: List[Tuple[str, str, str]]
    confidence: float = 0.0
    support: int = 0
    coverage: float = 0.0

    def __repr__(self):
        body_str = " AND ".join([f"{s}->{p}->{o}" for s, p, o in self.body])
        return f"{self.head[0]}->{self.head[1]}->{self.head[2]} <= ({body_str})"


@dataclass
class QueryPattern:
    """
    Represents a query pattern for complex query answering.

    Attributes:
        variables: Set of variable names
        predicates: List of (subject, relation, object) with variables
        target_var: Variable to predict
    """

    variables: Set[str]
    predicates: List[Tuple[str, str, str]]
    target_var: str


class RuleMining:
    """
    Rule mining for discovering Horn rules from knowledge graphs.

    Implements association rule mining adapted for knowledge graphs.
    """

    def __init__(
        self,
        min_support: int = 3,
        min_confidence: float = 0.1,
        max_rule_length: int = 3,
    ):
        self.min_support = min_support
        self.min_confidence = min_confidence
        self.max_rule_length = max_rule_length

    def mine_rules(
        self,
        triplets: List[Tuple[str, str, str]],
        relation_of_interest: Optional[str] = None,
    ) -> List[HornRule]:
        """
        Mine Horn rules from triplets.

        Args:
            triplets: List of (subject, predicate, object) triplets
            relation_of_interest: Only mine rules for this relation as head

        Returns:
            List of discovered Horn rules
        """
        from collections import Counter

        triplet_set = set(triplets)

        head_relations = [t[1] for t in triplets]
        if relation_of_interest:
            head_relations = [relation_of_interest]

        rules = []

        for head_rel in set(head_relations):
            head_triplets = [(s, p, o) for s, p, o in triplets if p == head_rel]

            for head in head_triplets:
                for length in range(1, self.max_rule_length + 1):
                    candidates = self._generate_candidate_bodies(
                        head, triplet_set, length
                    )

                    for body in candidates:
                        support = self._compute_support(head, body, triplet_set)

                        if support >= self.min_support:
                            confidence = support / len(head_triplets)

                            if confidence >= self.min_confidence:
                                rule = HornRule(
                                    head=head,
                                    body=body,
                                    confidence=confidence,
                                    support=support,
                                    coverage=support / len(head_triplets),
                                )
                                rules.append(rule)

        return sorted(rules, key=lambda r: r.confidence, reverse=True)

    def _generate_candidate_bodies(
        self,
        head: Tuple[str, str, str],
        triplet_set: Set[Tuple[str, str, str]],
        length: int,
    ) -> List[List[Tuple[str, str, str]]]:
        candidates = []

        head_s, head_r, head_o = head

        intermediate_entities = set()

        for s, p, o in triplet_set:
            if s == head_s or o == head_s:
                intermediate_entities.add(o if s == head_s else s)
            if s == head_o or o == head_o:
                intermediate_entities.add(o if s == head_o else s)

        for e in intermediate_entities:
            if e != head_s and e != head_o:
                body1 = [(head_s, "intermediate_rel", e)]
                if self._check_body_validity(body1, triplet_set, head_s, head_o):
                    candidates.append(body1)

                for e2 in intermediate_entities:
                    if e != e2:
                        body2 = [
                            (head_s, "r1", e),
                            (e, "r2", e2),
                        ]
                        if self._check_body_validity(
                            body2, triplet_set, head_s, head_o
                        ):
                            candidates.append(body2)

        return candidates

    def _check_body_validity(
        self,
        body: List[Tuple[str, str, str]],
        triplet_set: Set[Tuple[str, str, str]],
        head_s: str,
        head_o: str,
    ) -> bool:
        return True

    def _compute_support(
        self,
        head: Tuple[str, str, str],
        body: List[Tuple[str, str, str]],
        triplet_set: Set[Tuple[str, str, str]],
    ) -> int:
        head_s, head_r, head_o = head

        grounding_count = 0

        entities = list(set([head_s, head_o] + [e for t in body for e in [t[0], t[2]]]))

        for e in entities:
            grounded_body = []
            valid = True

            for s, p, o in body:
                new_s = e if s == "?" else s
                new_o = e if o == "?" else o

                if (new_s, p, new_o) not in triplet_set:
                    valid = False
                    break

            if valid:
                grounding_count += 1

        return grounding_count


class PathRanking:
    """
    Path-based relation reasoning using path ranking algorithm.

    Reference: Lao et al., Random Walk Inference and Learning in Large Knowledge Bases
    """

    def __init__(self, max_path_length: int = 3):
        self.max_path_length = max_path_length
        self.path_features: Dict[Tuple[str, str], List[Tensor]] = {}
        self.path_types: Dict[str, List[str]] = {}

    def extract_paths(
        self,
        graph: Dict[str, List[Tuple[str, str, float]]],
        source: str,
        target: str,
        max_length: Optional[int] = None,
    ) -> List[List[str]]:
        """
        Extract all paths between source and target.

        Args:
            graph: Adjacency list representation
            source: Source entity
            target: Target entity
            max_length: Maximum path length

        Returns:
            List of paths (each path is list of relations)
        """
        max_length = max_length or self.max_path_length

        paths = []

        def dfs(current: str, path: List[str], visited: Set[str]):
            if current == target:
                paths.append(path.copy())
                return

            if len(path) >= max_length:
                return

            for neighbor, relation, weight in graph.get(current, []):
                if neighbor not in visited:
                    visited.add(neighbor)
                    path.append(relation)
                    dfs(neighbor, path, visited)
                    path.pop()
                    visited.remove(neighbor)

        dfs(source, [], {source})

        return paths

    def compute_path_features(
        self,
        triplets: List[Tuple[str, str, str]],
    ) -> Dict[Tuple[str, str], Tensor]:
        """
        Compute path features for all entity pairs.

        Args:
            triplets: List of triplets

        Returns:
            Dictionary mapping (source, target) to path feature vector
        """
        graph = defaultdict(list)

        for s, p, o in triplets:
            graph[s].append((o, p, 1.0))

        features = {}

        for s in graph:
            for o in graph:
                if s != o:
                    paths = self.extract_paths(dict(graph), s, o)

                    if paths:
                        path_counts = defaultdict(int)
                        for path in paths:
                            path_key = tuple(path)
                            path_counts[path_key] += 1

                        features[(s, o)] = torch.tensor(list(path_counts.values()))

        return features

    def rank_relations(
        self,
        source: str,
        target: str,
        triplets: List[Tuple[str, str, str]],
        top_k: int = 10,
    ) -> List[Tuple[str, float]]:
        """
        Rank relations between source and target.

        Args:
            source: Source entity
            target: Target entity
            triplets: List of triplets
            top_k: Number of top relations to return

        Returns:
            List of (relation, score) tuples
        """
        graph = defaultdict(list)

        for s, p, o in triplets:
            graph[s].append((o, p, 1.0))

        paths = self.extract_paths(dict(graph), source, target)

        relation_scores = defaultdict(float)

        for path in paths:
            for rel in path:
                relation_scores[rel] += 1.0

        sorted_relations = sorted(
            relation_scores.items(), key=lambda x: x[1], reverse=True
        )

        return sorted_relations[:top_k]


class LogicalInference:
    """
    Logical inference engine using forward and backward chaining.

    Supports Horn clause reasoning and query answering.
    """

    def __init__(self):
        self.facts: Set[Tuple[str, str, str]] = set()
        self.rules: List[HornRule] = []
        self.inferred_facts: Set[Tuple[str, str, str]] = set()

    def add_fact(self, subject: str, predicate: str, obj: str) -> None:
        """Add a fact to the knowledge base."""
        self.facts.add((subject, predicate, obj))

    def add_rule(self, rule: HornRule) -> None:
        """Add a Horn rule to the knowledge base."""
        self.rules.append(rule)

    def forward_chain(self, max_iterations: int = 100) -> Set[Tuple[str, str, str]]:
        """
        Perform forward chaining inference.

        Args:
            max_iterations: Maximum number of iterations

        Returns:
            Set of inferred facts
        """
        new_facts = self.facts.copy()

        for _ in range(max_iterations):
            newly_inferred = set()

            for rule in self.rules:
                head_s, head_r, head_o = rule.head

                for s, p, o in new_facts:
                    if p == "type" and o == "person":
                        continue

                for body in [rule.body]:
                    groundings = self._match_body(body, new_facts)

                    for grounding in groundings:
                        inferred = self._apply_rule(rule, grounding)

                        if inferred not in new_facts and inferred not in newly_inferred:
                            newly_inferred.add(inferred)

            if not newly_inferred:
                break

            new_facts |= newly_inferred

        self.inferred_facts = new_facts - self.facts

        return self.inferred_facts

    def backward_chain(self, query: Tuple[str, str, str]) -> bool:
        """
        Perform backward chaining to answer a query.

        Args:
            query: Query triplet (subject, predicate, object)

        Returns:
            True if query can be proven
        """
        if query in self.facts:
            return True

        if query in self.inferred_facts:
            return True

        for rule in self.rules:
            if rule.head == query:
                if self._prove_body(rule.body, set()):
                    return True

        return False

    def _match_body(
        self,
        body: List[Tuple[str, str, str]],
        facts: Set[Tuple[str, str, str]],
    ) -> List[Dict[str, str]]:
        """Match body against known facts."""
        if not body:
            return [{}]

        groundings = []
        first_pred = body[0]

        for fact in facts:
            if self._unify(first_pred, fact, {}):
                groundings.append({first_pred[0]: fact[0], first_pred[2]: fact[2]})

        return groundings

    def _unify(
        self,
        pattern: Tuple[str, str, str],
        fact: Tuple[str, str, str],
        bindings: Dict[str, str],
    ) -> bool:
        """Unify pattern with fact given bindings."""
        p_s, p_p, p_o = pattern
        f_s, f_p, f_o = fact

        if p_p != f_p:
            return False

        if p_s in bindings and bindings[p_s] != f_s:
            return False
        if p_o in bindings and bindings[p_o] != f_o:
            return False

        return True

    def _prove_body(
        self,
        body: List[Tuple[str, str, str]],
        bindings: Dict[str, str],
    ) -> bool:
        """Prove body using backward chaining."""
        if not body:
            return True

        first = body[0]

        relevant_facts = [
            f for f in self.facts | self.inferred_facts if f[1] == first[1]
        ]

        for fact in relevant_facts:
            new_bindings = bindings.copy()

            if self._unify(first, fact, new_bindings):
                if self._prove_body(body[1:], new_bindings):
                    return True

        return False

    def _apply_rule(
        self,
        rule: HornRule,
        grounding: Dict[str, str],
    ) -> Tuple[str, str, str]:
        """Apply rule with given grounding."""
        h_s, h_r, h_o = rule.head

        s = grounding.get(h_s, h_s)
        o = grounding.get(h_o, h_o)

        return (s, h_r, o)

    def answer_conjunction_query(
        self,
        queries: List[Tuple[str, str, str]],
    ) -> List[Tuple[str, str]]:
        """
        Answer conjunction queries.

        Args:
            queries: List of (var1, relation, var2) patterns

        Returns:
            List of (var1, var2) tuples that satisfy all queries
        """
        if not queries:
            return []

        candidates = set()

        first_query = queries[0]
        for fact in self.facts | self.inferred_facts:
            if fact[1] == first_query[1]:
                candidates.add((fact[0], fact[2]))

        results = []

        for s, o in candidates:
            all_match = True

            for query in queries:
                s_match = s == query[0] or query[0] == "?"
                o_match = o == query[2] or query[2] == "?"

                if not (s_match and o_match):
                    all_match = False
                    break

                if not ((s, query[1], o) in (self.facts | self.inferred_facts)):
                    all_match = False
                    break

            if all_match:
                results.append((s, o))

        return results


class QueryEmbedding(nn.Module):
    """
    Query embedding for complex logical queries.

    Implements compositional query answering using logical operators.
    """

    def __init__(
        self,
        num_entities: int,
        num_relations: int,
        embedding_dim: int = 128,
        query_types: Optional[List[str]] = None,
    ):
        super().__init__()

        self.num_entities = num_entities
        self.num_relations = num_relations
        self.embedding_dim = embedding_dim

        self.query_types = query_types or [
            "projection",
            "intersection",
            "union",
            "negation",
        ]

        self.entity_embeddings = nn.Embedding(num_entities, embedding_dim)
        self.relation_embeddings = nn.Embedding(num_relations, embedding_dim)

        self.projection_op = nn.Linear(embedding_dim, embedding_dim)
        self.intersection_op = nn.Linear(embedding_dim, embedding_dim)
        self.union_op = nn.Linear(embedding_dim, embedding_dim)

    def forward(
        self,
        query_tokens: Tensor,
        query_type: str,
    ) -> Tensor:
        """
        Process a query and return scores for all entities.

        Args:
            query_tokens: Token IDs for the query
            query_type: Type of query operation

        Returns:
            Scores for all entities
        """
        entity_embeds = self.entity_embeddings.weight

        if query_type == "projection":
            return self._projection_query(query_tokens, entity_embeds)
        elif query_type == "intersection":
            return self._intersection_query(query_tokens, entity_embeds)
        elif query_type == "union":
            return self._union_query(query_tokens, entity_embeds)
        else:
            return self._projection_query(query_tokens, entity_embeds)

    def _projection_query(
        self,
        query_tokens: Tensor,
        entity_embeds: Tensor,
    ) -> Tensor:
        scores = self.projection_op(entity_embeds)
        return torch.sum(scores, dim=-1)

    def _intersection_query(
        self,
        query_tokens: Tensor,
        entity_embeds: Tensor,
    ) -> Tensor:
        embedded = self.entity_embeddings(query_tokens)
        pooled = torch.mean(embedded, dim=0)
        scores = self.intersection_op(pooled)
        return torch.matmul(scores, entity_embeds.T)

    def _union_query(
        self,
        query_tokens: Tensor,
        entity_embeds: Tensor,
    ) -> Tensor:
        embedded = self.entity_embeddings(query_tokens)
        pooled = torch.mean(embedded, dim=0)
        scores = self.union_op(pooled)
        return torch.matmul(scores, entity_embeds.T)
