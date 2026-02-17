"""
Neural-Symbolic Reasoning Framework.

Implements:
- Neural-symbolic architecture base
- Knowledge graph embeddings
- Relational reasoning
- Logical constraint satisfaction
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Dict, Set, Tuple, Callable, Any
import torch
from torch import Tensor, nn
import torch.nn.functional as F
import numpy as np
from collections import defaultdict


@dataclass
class KnowledgeBase:
    """Knowledge base with facts and rules.

    Attributes:
        facts: Set of ground atoms (tuples of predicate and arguments)
        rules: List of first-order rules
        predicates: Dictionary of predicate definitions
    """

    facts: Set[Tuple[str, Tuple[str, ...]]] = field(default_factory=set)
    rules: List[
        Tuple[Set[Tuple[str, Tuple[str, ...]]], Tuple[str, Tuple[str, ...]]]
    ] = field(default_factory=list)
    predicates: Dict[str, int] = field(default_factory=dict)

    def add_fact(self, predicate: str, args: Tuple[str, ...]) -> None:
        """Add a ground fact."""
        self.facts.add((predicate, args))

    def add_rule(
        self,
        antecedent: Set[Tuple[str, Tuple[str, ...]]],
        consequent: Tuple[str, Tuple[str, ...]],
    ) -> None:
        """Add a rule (antecedent -> consequent)."""
        self.rules.append((antecedent, consequent))

    def query(self, predicate: str, args: Tuple[str, ...]) -> bool:
        """Query the knowledge base."""
        return (predicate, args) in self.facts

    def forward_chain(self) -> Set[Tuple[str, Tuple[str, ...]]]:
        """Apply forward chaining to derive new facts."""
        new_facts = self.facts.copy()

        changed = True
        while changed:
            changed = False
            for antecedent, consequent in self.rules:
                if antecedent.issubset(new_facts):
                    if consequent not in new_facts:
                        new_facts.add(consequent)
                        changed = True

        return new_facts


class NeuralKnowledgeBase(nn.Module):
    """Neural knowledge base with embeddings.

    Combines symbolic knowledge with neural embeddings.
    """

    def __init__(
        self,
        n_entities: int,
        n_relations: int,
        embedding_dim: int = 64,
        n_rules: int = 0,
    ):
        """Initialize neural knowledge base.

        Args:
            n_entities: Number of entities
            n_relations: Number of relations
            embedding_dim: Dimension of embeddings
            n_rules: Number of rules to incorporate
        """
        super().__init__()
        self.n_entities = n_entities
        self.n_relations = n_relations
        self.embedding_dim = embedding_dim
        self.n_rules = n_rules

        self.entity_embeddings = nn.Embedding(n_entities, embedding_dim)
        self.relation_embeddings = nn.Embedding(n_relations, embedding_dim)

        if n_rules > 0:
            self.rule_encoder = nn.Sequential(
                nn.Linear(embedding_dim * 3, embedding_dim),
                nn.ReLU(),
                nn.Linear(embedding_dim, 1),
            )

    def forward(
        self,
        heads: Tensor,
        relations: Tensor,
        tails: Tensor,
    ) -> Tensor:
        """Score triples.

        Args:
            heads: Head entity IDs [batch]
            relations: Relation IDs [batch]
            tails: Tail entity IDs [batch]

        Returns:
            Scores [batch]
        """
        head_embeds = self.entity_embeddings(heads)
        rel_embeds = self.relation_embeddings(relations)
        tail_embeds = self.entity_embeddings(tails)

        score = (head_embeds + rel_embeds - tail_embeds).pow(2).sum(dim=-1)

        return -score

    def get_entity_embedding(self, entity_id: int) -> Tensor:
        """Get embedding for an entity."""
        return self.entity_embeddings(torch.tensor(entity_id))

    def get_relation_embedding(self, relation_id: int) -> Tensor:
        """Get embedding for a relation."""
        return self.relation_embeddings(torch.tensor(relation_id))


class ComplEx(nn.Module):
    """Complex embeddings for knowledge graphs.

    From: "Complex Embeddings for Simple Link Prediction"
    """

    def __init__(
        self,
        n_entities: int,
        n_relations: int,
        embedding_dim: int = 64,
    ):
        """Initialize ComplEx model.

        Args:
            n_entities: Number of entities
            n_relations: Number of relations
            embedding_dim: Embedding dimension
        """
        super().__init__()
        self.n_entities = n_entities
        self.n_relations = n_relations
        self.embedding_dim = embedding_dim

        self.entity_real = nn.Embedding(n_entities, embedding_dim)
        self.entity_imag = nn.Embedding(n_entities, embedding_dim)
        self.relation_real = nn.Embedding(n_relations, embedding_dim)
        self.relation_imag = nn.Embedding(n_relations, embedding_dim)

    def forward(
        self,
        heads: Tensor,
        relations: Tensor,
        tails: Tensor,
    ) -> Tensor:
        """Score triples using ComplEx.

        Args:
            heads: Head entity IDs [batch]
            relations: Relation IDs [batch]
            tails: Tail entity IDs [batch]

        Returns:
            Scores [batch]
        """
        h_real = self.entity_real(heads)
        h_imag = self.entity_imag(heads)
        r_real = self.relation_real(relations)
        r_imag = self.relation_imag(relations)
        t_real = self.entity_real(tails)
        t_imag = self.entity_imag(tails)

        score_real = (
            h_real * r_real * t_real
            + h_imag * r_real * t_imag
            + h_real * r_imag * t_imag
            - h_imag * r_imag * t_real
        )
        score_imag = (
            h_real * r_real * t_imag
            - h_imag * r_real * t_real
            + h_real * r_imag * t_real
            + h_imag * r_imag * t_imag
        )

        score = score_real + score_imag

        return score.sum(dim=-1)


class RelationalReasoning(nn.Module):
    """Relational reasoning over graph structures.

    Implements message passing for logical reasoning.
    """

    def __init__(
        self,
        node_dim: int,
        edge_dim: int,
        hidden_dim: int = 64,
        n_layers: int = 3,
        message_fn: str = "mlp",
    ):
        """Initialize relational reasoning.

        Args:
            node_dim: Node feature dimension
            edge_dim: Edge feature dimension
            hidden_dim: Hidden dimension
            n_layers: Number of message passing layers
            message_fn: Message function type ("mlp", "dot")
        """
        super().__init__()
        self.node_dim = node_dim
        self.edge_dim = edge_dim
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers
        self.message_fn = message_fn

        self.node_encoder = nn.Linear(node_dim, hidden_dim)
        self.edge_encoder = nn.Linear(edge_dim, hidden_dim)

        if message_fn == "mlp":
            self.message_mlp = nn.Sequential(
                nn.Linear(hidden_dim * 3, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim),
            )
        elif message_fn == "dot":
            pass

        self.update_mlp = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

    def forward(
        self,
        node_features: Tensor,
        edge_features: Tensor,
        edge_index: Tensor,
    ) -> Tensor:
        """Perform relational reasoning.

        Args:
            node_features: Node features [n_nodes, node_dim]
            edge_features: Edge features [n_edges, edge_dim]
            edge_index: Edge connectivity [2, n_edges]

        Returns:
            Updated node features [n_nodes, hidden_dim]
        """
        x = self.node_encoder(node_features)
        e = self.edge_encoder(edge_features)

        for _ in range(self.n_layers):
            messages = self._compute_messages(x, e, edge_index)
            x = self._update_nodes(x, messages, edge_index)

        return x

    def _compute_messages(
        self,
        x: Tensor,
        e: Tensor,
        edge_index: Tensor,
    ) -> Tensor:
        """Compute messages along edges."""
        src, dst = edge_index

        if self.message_fn == "mlp":
            message_input = torch.cat([x[src], x[dst], e], dim=-1)
            messages = self.message_mlp(message_input)
        elif self.message_fn == "dot":
            messages = x[src] * x[dst] * e

        return messages

    def _update_nodes(
        self,
        x: Tensor,
        messages: Tensor,
        edge_index: Tensor,
    ) -> Tensor:
        """Aggregate messages and update node features."""
        src, dst = edge_index

        n_nodes = x.size(0)
        aggregated = torch.zeros(n_nodes, self.hidden_dim, device=x.device)

        aggregated.index_add_(0, dst, messages)

        combined = torch.cat([x, aggregated], dim=-1)
        x_new = self.update_mlp(combined)

        return x_new + x


class LogicalConstraintSatisfaction(nn.Module):
    """Neural network with logical constraint satisfaction.

    Ensures outputs satisfy given logical constraints.
    """

    def __init__(
        self,
        base_network: nn.Module,
        constraints: List[Callable[[Tensor], Tensor]],
        constraint_weights: Optional[List[float]] = None,
        penalty_weight: float = 1.0,
    ):
        """Initialize constrained network.

        Args:
            base_network: Base neural network
            constraints: List of constraint functions (return penalty)
            constraint_weights: Weights for each constraint
            penalty_weight: Overall penalty scaling
        """
        super().__init__()
        self.base_network = base_network
        self.constraints = constraints
        self.constraint_weights = constraint_weights or [1.0] * len(constraints)
        self.penalty_weight = penalty_weight

    def forward(self, x: Tensor) -> Tuple[Tensor, Dict[str, Tensor]]:
        """Forward pass with constraint satisfaction.

        Args:
            x: Input

        Returns:
            Tuple of (output, info_dict)
        """
        output = self.base_network(x)

        total_penalty = torch.tensor(0.0, device=x.device)
        penalties = {}

        for i, (constraint, weight) in enumerate(
            zip(self.constraints, self.constraint_weights)
        ):
            penalty = constraint(output)
            penalties[f"constraint_{i}"] = penalty
            total_penalty = total_penalty + weight * penalty

        info = {
            "output": output,
            "constraint_penalty": total_penalty,
            "penalties": penalties,
        }

        return output, info

    def get_loss(self, x: Tensor, target: Tensor, task_loss_fn: Callable) -> Tensor:
        """Compute loss including constraint penalties.

        Args:
            x: Input
            target: Target
            task_loss_fn: Task-specific loss function

        Returns:
            Total loss
        """
        output, info = self.forward(x)
        task_loss = task_loss_fn(output, target)
        total_loss = task_loss + self.penalty_weight * info["constraint_penalty"]
        return total_loss


class NeuroSymbolicLayer(nn.Module):
    """Generic neuro-symbolic layer.

    Combines neural processing with symbolic operations.
    """

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        symbolic_ops: List[str],
        hidden_dim: int = 64,
    ):
        """Initialize neuro-symbolic layer.

        Args:
            input_dim: Input dimension
            output_dim: Output dimension
            symbolic_ops: List of symbolic operations to apply
            hidden_dim: Hidden dimension
        """
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.symbolic_ops = symbolic_ops
        self.hidden_dim = hidden_dim

        self.neural_encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

        self.symbolic_projections = nn.ModuleDict()
        for op in symbolic_ops:
            self.symbolic_projections[op] = nn.Linear(hidden_dim, output_dim)

        self.combiner = nn.Sequential(
            nn.Linear(output_dim * (len(symbolic_ops) + 1), hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, x: Tensor) -> Tensor:
        """Forward pass.

        Args:
            x: Input [batch, input_dim]

        Returns:
            Output [batch, output_dim]
        """
        encoded = self.neural_encoder(x)

        outputs = [encoded]

        for op in self.symbolic_ops:
            proj = self.symbolic_projections[op](encoded)
            outputs.append(proj)

        combined = torch.cat(outputs, dim=-1)
        return self.combiner(combined)


class NeuralTheoremProverInput:
    """Input for neural theorem proving."""

    def __init__(
        self,
        premises: List[str],
        hypothesis: str,
    ):
        """Initialize theorem proving input.

        Args:
            premises: List of premise formulas
            hypothesis: Hypothesis to prove
        """
        self.premises = premises
        self.hypothesis = hypothesis


class EntailmentGrounder:
    """Grounds logical formulas to neural embeddings.

    Converts symbolic formulas into tensor operations.
    """

    def __init__(
        self,
        embedding_dim: int = 64,
    ):
        """Initialize entailment grounder.

        Args:
            embedding_dim: Dimension of embeddings
        """
        self.embedding_dim = embedding_dim

        self.formula_encoder = nn.Sequential(
            nn.Linear(embedding_dim * 3, embedding_dim),
            nn.ReLU(),
            nn.Linear(embedding_dim, embedding_dim),
        )

    def ground_atom(self, predicate: str, args: List[Tensor]) -> Tensor:
        """Ground an atom (predicate applied to arguments).

        Args:
            predicate: Predicate name
            args: Argument embeddings

        Returns:
            Grounding tensor
        """
        if len(args) == 1:
            return args[0]
        elif len(args) == 2:
            return torch.cat(args, dim=-1)
        return torch.cat(args, dim=-1)

    def ground_conjunction(self, conjuncts: List[Tensor]) -> Tensor:
        """Ground conjunction (AND).

        Args:
            conjuncts: List of grounded conjuncts

        Returns:
            Conjunction grounding
        """
        return torch.stack(conjuncts).prod(dim=0)

    def ground_disjunction(self, disjuncts: List[Tensor]) -> Tensor:
        """Ground disjunction (OR).

        Args:
            disjuncts: List of grounded disjuncts

        Returns:
            Disjunction grounding
        """
        return 1.0 - torch.stack([1.0 - d for d in disjuncts]).prod(dim=0)

    def ground_implication(self, antecedent: Tensor, consequent: Tensor) -> Tensor:
        """Ground implication (A -> B).

        Args:
            antecedent: Antecedent grounding
            consequent: Consequent grounding

        Returns:
            Implication grounding
        """
        return torch.max(1.0 - antecedent, consequent)


class SemanticLoss(nn.Module):
    """Semantic loss for logical constraints.

    From: "Semantic Loss Functions for Neuro-Symbolic Learning"
    """

    def __init__(
        self,
        formula: Callable[[Tensor], Tensor],
        t_norm: str = "product",
    ):
        """Initialize semantic loss.

        Args:
            formula: Formula that returns satisfaction degree
            t_norm: T-norm for soft logic
        """
        super().__init__()
        self.formula = formula
        self.t_norm = t_norm

    def forward(self, predictions: Tensor, target: Tensor) -> Tensor:
        """Compute semantic loss.

        Args:
            predictions: Network predictions
            target: Target truth values

        Returns:
            Loss value
        """
        satisfaction = self.formula(predictions)

        target_satisfaction = self._compute_target_satisfaction(target)

        loss = F.mse_loss(satisfaction, target_satisfaction)

        return loss

    def _compute_target_satisfaction(self, target: Tensor) -> Tensor:
        """Compute what satisfaction the target should have."""
        return target


class KnowledgeGraphReasoner(nn.Module):
    """Complete knowledge graph reasoning system.

    Combines embedding-based and rule-based reasoning.
    """

    def __init__(
        self,
        n_entities: int,
        n_relations: int,
        embedding_dim: int = 64,
        use_rules: bool = True,
    ):
        """Initialize knowledge graph reasoner.

        Args:
            n_entities: Number of entities
            n_relations: Number of relations
            embedding_dim: Embedding dimension
            use_rules: Whether to use rule-based reasoning
        """
        super().__init__()
        self.n_entities = n_entities
        self.n_relations = n_relations
        self.embedding_dim = embedding_dim
        self.use_rules = use_rules

        self.embedding_model = NeuralKnowledgeBase(
            n_entities, n_relations, embedding_dim
        )

        if use_rules:
            self.rule_attention = nn.MultiheadAttention(
                embedding_dim, num_heads=4, batch_first=True
            )

    def forward(
        self,
        queries: Tensor,
    ) -> Dict[str, Tensor]:
        """Answer queries.

        Args:
            queries: Query triplets [batch, 3] (head, relation, ?)

        Returns:
            Predictions and scores
        """
        heads = queries[:, 0]
        relations = queries[:, 1]
        tails = queries[:, 2]

        all_scores = []
        for tail_id in range(self.n_entities):
            tail_tensor = torch.full((heads.size(0),), tail_id, device=heads.device)
            scores = self.embedding_model(heads, relations, tail_tensor)
            all_scores.append(scores)

        all_scores = torch.stack(all_scores, dim=1)

        return {
            "scores": all_scores,
            "predictions": all_scores.argmax(dim=-1),
        }

    def link_prediction(
        self,
        head: int,
        relation: int,
        tail: Optional[int] = None,
    ) -> List[Tuple[int, float]]:
        """Perform link prediction.

        Args:
            head: Head entity
            relation: Relation
            tail: Optional tail entity (if known)

        Returns:
            List of (entity, score) pairs
        """
        head_tensor = torch.tensor(
            [head], device=self.embedding_model.entity_embeddings.weight.device
        )
        rel_tensor = torch.tensor(
            [relation], device=self.embedding_model.relation_embeddings.weight.device
        )

        scores = []
        for tail_id in range(self.n_entities):
            tail_tensor = torch.tensor(
                [tail_id], device=self.embedding_model.entity_embeddings.weight.device
            )
            score = self.embedding_model(head_tensor, rel_tensor, tail_tensor)
            scores.append((tail_id, score.item()))

        scores.sort(key=lambda x: x[1], reverse=True)
        return scores
