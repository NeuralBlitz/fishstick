"""
Neuro-Symbolic Hybrid Architectures.

Implements:
- Neural-symbolic integration patterns
- Differentiable inductive programming
- Lifted neural networks
- Graph neural networks for logical structures
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Dict, Set, Tuple, Callable, Any
import torch
from torch import Tensor, nn
import torch.nn.functional as F
import numpy as np
from collections import defaultdict


class LiftedNeuralNetwork(nn.Module):
    """Lifted neural network that operates on sets and relations.

    Applies the same neural network to each element of a set or relation.
    """

    def __init__(
        self,
        element_encoder: nn.Module,
        aggregator: str = "sum",
        n_layers: int = 2,
        hidden_dim: int = 64,
    ):
        """Initialize lifted neural network.

        Args:
            element_encoder: Encoder for individual elements
            aggregator: Aggregation function ("sum", "mean", "max", "attention")
            n_layers: Number of processing layers
            hidden_dim: Hidden dimension
        """
        super().__init__()
        self.element_encoder = element_encoder
        self.aggregator = aggregator
        self.hidden_dim = hidden_dim

        if aggregator == "attention":
            self.query_proj = nn.Linear(hidden_dim, hidden_dim)
            self.key_proj = nn.Linear(hidden_dim, hidden_dim)
            self.value_proj = nn.Linear(hidden_dim, hidden_dim)

        self.processing_layers = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(hidden_dim, hidden_dim),
                    nn.ReLU(),
                    nn.Linear(hidden_dim, hidden_dim),
                )
                for _ in range(n_layers)
            ]
        )

    def forward(self, elements: Tensor, mask: Optional[Tensor] = None) -> Tensor:
        """Process set of elements.

        Args:
            elements: Element features [batch, n_elements, element_dim]
            mask: Optional mask for valid elements [batch, n_elements]

        Returns:
            Set representation [batch, hidden_dim]
        """
        batch_size, n_elements, _ = elements.shape

        encoded = self.element_encoder(elements)

        for layer in self.processing_layers:
            encoded = layer(encoded) + encoded

        if self.aggregator == "sum":
            aggregated = encoded.sum(dim=1)
        elif self.aggregator == "mean":
            aggregated = encoded.mean(dim=1)
        elif self.aggregator == "max":
            aggregated = encoded.max(dim=1)[0]
        elif self.aggregator == "attention":
            aggregated = self._attention_aggregate(encoded, mask)
        else:
            raise ValueError(f"Unknown aggregator: {self.aggregator}")

        return aggregated

    def _attention_aggregate(
        self,
        encoded: Tensor,
        mask: Optional[Tensor],
    ) -> Tensor:
        """Attention-based aggregation."""
        batch_size, n_elements, hidden_dim = encoded.shape

        query = self.query_proj(encoded.mean(dim=1, keepdim=True))
        key = self.key_proj(encoded)
        value = self.value_proj(encoded)

        scores = torch.matmul(query, key.transpose(-2, -1)) / (hidden_dim**0.5)

        if mask is not None:
            scores = scores.masked_fill(~mask.unsqueeze(1), float("-inf"))

        attn_weights = F.softmax(scores, dim=-1)

        aggregated = torch.matmul(attn_weights, value).squeeze(1)

        return aggregated


class RelationalNeuralNetwork(nn.Module):
    """Neural network for relational data.

    Processes entities and their relations.
    """

    def __init__(
        self,
        entity_dim: int,
        relation_dim: int,
        hidden_dim: int = 64,
        n_layers: int = 3,
        message_passing: bool = True,
    ):
        """Initialize relational neural network.

        Args:
            entity_dim: Entity feature dimension
            relation_dim: Relation feature dimension
            hidden_dim: Hidden dimension
            n_layers: Number of layers
            message_passing: Whether to use message passing
        """
        super().__init__()
        self.entity_dim = entity_dim
        self.relation_dim = relation_dim
        self.hidden_dim = hidden_dim
        self.message_passing = message_passing

        self.entity_encoder = nn.Linear(entity_dim, hidden_dim)
        self.relation_encoder = nn.Linear(relation_dim, hidden_dim)

        if message_passing:
            self.message_mlp = nn.Sequential(
                nn.Linear(hidden_dim * 3, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim),
            )

            self.update_mlp = nn.Sequential(
                nn.Linear(hidden_dim * 2, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim),
            )

        self.output_mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(
        self,
        entity_features: Tensor,
        relation_features: Tensor,
        edge_index: Tensor,
    ) -> Tensor:
        """Process relational data.

        Args:
            entity_features: Entity features [n_entities, entity_dim]
            relation_features: Relation features [n_edges, relation_dim]
            edge_index: Edge connectivity [2, n_edges]

        Returns:
            Entity representations [n_entities, hidden_dim]
        """
        x = self.entity_encoder(entity_features)
        r = self.relation_encoder(relation_features)

        if self.message_passing:
            for _ in range(3):
                messages = self._compute_messages(x, r, edge_index)
                x = self._update_entities(x, messages, edge_index)

        return x

    def _compute_messages(
        self,
        x: Tensor,
        r: Tensor,
        edge_index: Tensor,
    ) -> Tensor:
        """Compute messages along edges."""
        src, dst = edge_index

        message_input = torch.cat([x[src], r, x[dst]], dim=-1)
        messages = self.message_mlp(message_input)

        return messages

    def _update_entities(
        self,
        x: Tensor,
        messages: Tensor,
        edge_index: Tensor,
    ) -> Tensor:
        """Update entity features with messages."""
        src, dst = edge_index

        n_entities = x.size(0)
        aggregated = torch.zeros(n_entities, self.hidden_dim, device=x.device)

        aggregated.index_add_(0, dst, messages)

        combined = torch.cat([x, aggregated], dim=-1)
        x_new = self.update_mlp(combined)

        return x_new + x


class DifferentiableILP(nn.Module):
    """Differentiable Inductive Logic Programming.

    Learns logic programs with gradient-based optimization.
    """

    def __init__(
        self,
        n_atoms: int,
        n_rules: int,
        embedding_dim: int = 64,
    ):
        """Initialize differentiable ILP.

        Args:
            n_atoms: Number of atoms
            n_rules: Number of rules
            embedding_dim: Embedding dimension
        """
        super().__init__()
        self.n_atoms = n_atoms
        self.n_rules = n_rules
        self.embedding_dim = embedding_dim

        self.atom_embeddings = nn.Embedding(n_atoms, embedding_dim)
        self.rule_embeddings = nn.Embedding(n_rules, embedding_dim)

        self.rule_predictor = nn.Sequential(
            nn.Linear(embedding_dim * 2, embedding_dim),
            nn.ReLU(),
            nn.Linear(embedding_dim, n_rules),
        )

    def forward(
        self,
        atom_ids: Tensor,
    ) -> Dict[str, Tensor]:
        """Predict rules for atoms.

        Args:
            atom_ids: Atom IDs [batch, seq_len]

        Returns:
            Predictions
        """
        embeddings = self.atom_embeddings(atom_ids)

        atom_repr = embeddings.mean(dim=1)

        rule_scores = self.rule_predictor(atom_repr)

        return {
            "rule_scores": rule_scores,
            "atom_embeddings": embeddings,
        }

    def compute_ilp_loss(
        self,
        atom_ids: Tensor,
        target_rules: Tensor,
        pos_examples: Tensor,
        neg_examples: Tensor,
    ) -> Tensor:
        """Compute ILP loss.

        Args:
            atom_ids: Atom IDs
            target_rules: Target rule assignments
            pos_examples: Positive examples
            neg_examples: Negative examples

        Returns:
            Loss value
        """
        predictions = self.forward(atom_ids)

        rule_logits = predictions["rule_scores"]

        pos_loss = F.binary_cross_entropy_with_logits(
            rule_logits[pos_examples],
            target_rules[pos_examples],
        )

        neg_loss = F.binary_cross_entropy_with_logits(
            rule_logits[neg_examples],
            target_rules[neg_examples],
        )

        return pos_loss + neg_loss


class NeuralLogicMachine(nn.Module):
    """Neural Logic Machine (NLM).

    From: "Neural Logic Machines" (Liu et al., 2019)
    """

    def __init__(
        self,
        n_objects: int,
        n_predicates: int,
        embedding_dim: int = 64,
        n_layers: int = 3,
    ):
        """Initialize NLM.

        Args:
            n_objects: Number of objects
            n_predicates: Number of predicates
            embedding_dim: Embedding dimension
            n_layers: Number of layers
        """
        super().__init__()
        self.n_objects = n_objects
        self.n_predicates = n_predicates
        self.embedding_dim = embedding_dim
        self.n_layers = n_layers

        self.object_embeddings = nn.Embedding(n_objects, embedding_dim)

        self.predicate_layers = nn.ModuleList()
        for _ in range(n_layers):
            self.predicate_layers.append(
                nn.Sequential(
                    nn.Linear(embedding_dim * 3, embedding_dim),
                    nn.ReLU(),
                    nn.Linear(embedding_dim, n_predicates),
                    nn.Sigmoid(),
                )
            )

    def forward(
        self,
        object_ids: Tensor,
    ) -> Dict[str, Tensor]:
        """Forward pass.

        Args:
            object_ids: Object IDs [batch, n_objects]

        Returns:
            Predicate values
        """
        embeddings = self.object_embeddings(object_ids)

        batch_size, n_objects, _ = embeddings.shape

        predicates = []

        for layer in self.predicate_layers:
            predicate_values = []

            for i in range(n_objects):
                for j in range(n_objects):
                    if i != j:
                        triple = torch.cat(
                            [
                                embeddings[:, i, :],
                                embeddings[:, j, :],
                                embeddings.mean(dim=1),
                            ],
                            dim=-1,
                        )

                        pred = layer(triple)
                        predicate_values.append(pred)

            predicates.append(torch.stack(predicate_values, dim=1))

        return {
            "predicates": predicates,
            "embeddings": embeddings,
        }


class LogicGraphNetwork(nn.Module):
    """Graph neural network with logical constraints.

    Combines GNN message passing with logical reasoning.
    """

    def __init__(
        self,
        node_dim: int,
        edge_dim: int,
        hidden_dim: int = 64,
        n_layers: int = 3,
        use_logical_constraints: bool = True,
    ):
        """Initialize logic graph network.

        Args:
            node_dim: Node feature dimension
            edge_dim: Edge feature dimension
            hidden_dim: Hidden dimension
            n_layers: Number of GNN layers
            use_logical_constraints: Whether to apply logical constraints
        """
        super().__init__()
        self.node_dim = node_dim
        self.edge_dim = edge_dim
        self.hidden_dim = hidden_dim
        self.use_logical_constraints = use_logical_constraints

        self.node_encoder = nn.Linear(node_dim, hidden_dim)
        self.edge_encoder = nn.Linear(edge_dim, hidden_dim)

        self.message_layers = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(hidden_dim * 3, hidden_dim),
                    nn.ReLU(),
                    nn.Linear(hidden_dim, hidden_dim),
                )
                for _ in range(n_layers)
            ]
        )

        self.update_layers = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(hidden_dim * 2, hidden_dim),
                    nn.ReLU(),
                    nn.Linear(hidden_dim, hidden_dim),
                )
                for _ in range(n_layers)
            ]
        )

        if use_logical_constraints:
            self.constraint_mlp = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, 1),
                nn.Sigmoid(),
            )

    def forward(
        self,
        node_features: Tensor,
        edge_features: Tensor,
        edge_index: Tensor,
    ) -> Dict[str, Tensor]:
        """Forward pass.

        Args:
            node_features: Node features [n_nodes, node_dim]
            edge_features: Edge features [n_edges, edge_dim]
            edge_index: Edge connectivity [2, n_edges]

        Returns:
            Node representations and constraints
        """
        x = self.node_encoder(node_features)
        e = self.edge_encoder(edge_features)

        for msg_layer, update_layer in zip(self.message_layers, self.update_layers):
            messages = self._message_passing(x, e, edge_index, msg_layer)
            x = update_layer(torch.cat([x, messages], dim=-1))

        constraints = None
        if self.use_logical_constraints:
            constraints = self._compute_constraints(x)

        return {
            "node_representations": x,
            "constraints": constraints,
        }

    def _message_passing(
        self,
        x: Tensor,
        e: Tensor,
        edge_index: Tensor,
        msg_layer: nn.Module,
    ) -> Tensor:
        """Perform message passing."""
        src, dst = edge_index

        msg_input = torch.cat([x[src], e, x[dst]], dim=-1)
        messages = msg_layer(msg_input)

        n_nodes = x.size(0)
        aggregated = torch.zeros(n_nodes, self.hidden_dim, device=x.device)
        aggregated.index_add_(0, dst, messages)

        return aggregated

    def _compute_constraints(self, x: Tensor) -> Tensor:
        """Compute logical constraints."""
        constraint_values = self.constraint_mlp(x)

        symmetry = torch.matmul(x, x.T)

        return constraint_values


class SemanticParsingNetwork(nn.Module):
    """Neural network for semantic parsing.

    Converts natural language to logical forms.
    """

    def __init__(
        self,
        vocab_size: int,
        embed_dim: int = 128,
        hidden_dim: int = 256,
        n_layers: int = 2,
    ):
        """Initialize semantic parsing network.

        Args:
            vocab_size: Vocabulary size
            embed_dim: Embedding dimension
            hidden_dim: Hidden dimension
            n_layers: Number of layers
        """
        super().__init__()
        self.vocab_size = vocab_size
        self.embed_dim = embed_dim
        self.hidden_dim = hidden_dim

        self.embedding = nn.Embedding(vocab_size, embed_dim)

        self.encoder = nn.LSTM(
            embed_dim,
            hidden_dim,
            n_layers,
            batch_first=True,
            bidirectional=True,
        )

        self.decoder = nn.LSTM(
            embed_dim + hidden_dim * 2,
            hidden_dim,
            n_layers,
            batch_first=True,
        )

        self.output_proj = nn.Linear(hidden_dim, vocab_size)

    def forward(
        self,
        input_ids: Tensor,
        target_ids: Tensor,
    ) -> Dict[str, Tensor]:
        """Parse input to logical form.

        Args:
            input_ids: Input token IDs [batch, seq_len]
            target_ids: Target token IDs [batch, target_len]

        Returns:
            Predictions
        """
        input_embed = self.embedding(input_ids)
        target_embed = self.embedding(target_ids)

        encoder_output, (h_n, c_n) = self.encoder(input_embed)

        h_n = h_n.view(2, 2, -1)[-1]
        c_n = c_n.view(2, 2, -1)[-1]
        hidden = (h_n, c_n)

        decoder_output, _ = self.decoder(target_embed, hidden)

        logits = self.output_proj(decoder_output)

        return {
            "logits": logits,
            "encoder_output": encoder_output,
        }

    def generate(
        self,
        input_ids: Tensor,
        max_length: int = 50,
        temperature: float = 1.0,
    ) -> Tensor:
        """Generate logical form.

        Args:
            input_ids: Input token IDs
            max_length: Maximum generation length
            temperature: Sampling temperature

        Returns:
            Generated token IDs
        """
        input_embed = self.embedding(input_ids)
        encoder_output, (h_n, c_n) = self.encoder(input_embed)

        h_n = h_n.view(2, 2, -1)[-1]
        c_n = c_n.view(2, 2, -1)[-1]
        hidden = (h_n, c_n)

        generated = [1]
        current_id = torch.tensor([[1]], device=input_ids.device)

        for _ in range(max_length):
            target_embed = self.embedding(current_id)

            context = encoder_output.mean(dim=1, keepdim=True)
            decoder_input = torch.cat([target_embed, context], dim=-1)

            decoder_output, hidden = self.decoder(decoder_input, hidden)

            logits = self.output_proj(decoder_output[:, -1, :])

            probs = F.softmax(logits / temperature, dim=-1)
            next_token = probs.argmax(dim=-1)

            generated.append(next_token.item())
            current_id = next_token.unsqueeze(0)

            if next_token.item() == 2:
                break

        return torch.tensor([generated], device=input_ids.device)


class NeuroSymbolicIntegration(nn.Module):
    """Complete neuro-symbolic integration system.

    Combines neural and symbolic components.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 64,
        n_classes: int = 10,
    ):
        """Initialize neuro-symbolic integration.

        Args:
            input_dim: Input dimension
            hidden_dim: Hidden dimension
            n_classes: Number of output classes
        """
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.n_classes = n_classes

        self.neural_stream = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

        self.symbolic_encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

        self.rule_attention = nn.MultiheadAttention(
            hidden_dim, num_heads=4, batch_first=True
        )

        self.integration_mlp = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, n_classes),
        )

    def forward(
        self,
        x: Tensor,
        symbolic_features: Optional[Tensor] = None,
    ) -> Dict[str, Tensor]:
        """Forward pass with both streams.

        Args:
            x: Input features
            symbolic_features: Pre-computed symbolic features

        Returns:
            Predictions and intermediate representations
        """
        neural_out = self.neural_stream(x)

        if symbolic_features is None:
            symbolic_features = self.symbolic_encoder(x)

        attn_out, _ = self.rule_attention(
            neural_out.unsqueeze(1),
            symbolic_features.unsqueeze(1),
            symbolic_features.unsqueeze(1),
        )

        combined = torch.cat([neural_out, attn_out.squeeze(1)], dim=-1)

        output = self.integration_mlp(combined)

        return {
            "output": output,
            "neural_repr": neural_out,
            "symbolic_repr": symbolic_features,
        }
