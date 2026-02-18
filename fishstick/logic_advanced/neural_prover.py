import torch
import torch.nn as nn
from typing import List, Optional, Dict, Any, Tuple
from dataclasses import dataclass
from enum import Enum
import numpy as np


class LogicalConnective(Enum):
    AND = "and"
    OR = "or"
    NOT = "not"
    IMPLIES = "implies"
    IFF = "iff"


@dataclass
class Formula:
    connective: Optional[LogicalConnective]
    atoms: List[Any]
    negation: bool = False

    def __repr__(self) -> str:
        if self.negation:
            return f"¬{self._to_string()}"
        return self._to_string()

    def _to_string(self) -> str:
        if not self.connective:
            return str(self.atoms[0]) if self.atoms else "⊥"
        if self.connective == LogicalConnective.NOT:
            return f"¬{self.atoms[0]}"
        ops = {"and": "∧", "or": "∨", "implies": "→", "iff": "↔"}
        op = ops.get(self.connective.value, self.connective.value)
        if len(self.atoms) == 1:
            return f"{op}{self.atoms[0]}"
        return f"({f' {op} '.join(str(a) for a in self.atoms)})"


class TruthValue(Enum):
    TRUE = 1
    FALSE = 0
    UNKNOWN = 0.5


class NeuralProver(nn.Module):
    def __init__(
        self,
        vocab_size: int = 1000,
        embedding_dim: int = 256,
        hidden_dim: int = 512,
        num_layers: int = 6,
        num_heads: int = 8,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim

        self.token_embedding = nn.Embedding(vocab_size, embedding_dim)
        self.positional_embedding = nn.Embedding(512, embedding_dim)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embedding_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim,
            dropout=dropout,
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        self.formula_encoder = nn.Sequential(
            nn.Linear(embedding_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, embedding_dim),
        )

        self.proof_head = nn.Sequential(
            nn.Linear(embedding_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid(),
        )

        self.axiom_attention = nn.MultiheadAttention(
            embedding_dim, num_heads, dropout=dropout, batch_first=True
        )

    def forward(
        self,
        premises: torch.Tensor,
        hypotheses: torch.Tensor,
        premise_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        batch_size, num_premises, seq_len = premises.shape

        positions = (
            torch.arange(seq_len, device=premises.device)
            .unsqueeze(0)
            .expand(batch_size * num_premises, -1)
        )
        pos_emb = self.positional_embedding(positions)

        flat_premises = premises.view(-1, seq_len)
        premise_emb = self.token_embedding(flat_premises) + pos_emb

        premise_emb = premise_emb.view(batch_size, num_premises, seq_len, -1)
        premise_emb = premise_emb.mean(dim=2)

        premise_seq = premise_emb
        hypothesis_emb = self.token_embedding(hypotheses)

        transformed_premises, _ = self.axiom_attention(
            premise_seq, premise_seq, premise_seq, key_padding_mask=premise_mask
        )

        premise_pooled = transformed_premises.mean(dim=1, keepdim=True)
        premise_pooled = premise_pooled.expand(-1, hypothesis_emb.size(1), -1)

        combined = torch.cat([hypothesis_emb, premise_pooled], dim=-1)

        proof_score = self.proof_head(combined)
        return proof_score.squeeze(-1)

    def prove(
        self,
        premises: List[str],
        hypothesis: str,
        max_depth: int = 10,
    ) -> Tuple[bool, float, List[Formula]]:
        with torch.no_grad():
            premise_tokens = self._encode_formulas(premises)
            hypothesis_tokens = self._encode_formulas([hypothesis])

            scores = self.forward(premise_tokens, hypothesis_tokens)
            proof_score = scores.item()

            proof_found = proof_score > 0.5

            if proof_found:
                derivation = self._extract_derivation(premises, hypothesis)
                return True, proof_score, derivation
            return False, proof_score, []

    def _encode_formulas(self, formulas: List[str]) -> torch.Tensor:
        tokens = torch.randint(
            0,
            self.vocab_size,
            (len(formulas), 32),
            device=next(self.parameters()).device,
        )
        return tokens

    def _extract_derivation(
        self, premises: List[str], hypothesis: str
    ) -> List[Formula]:
        derivation = []
        for i, premise in enumerate(premises):
            formula = Formula(connective=None, atoms=[premise], negation=False)
            derivation.append(formula)
        target = Formula(connective=None, atoms=[hypothesis], negation=False)
        derivation.append(target)
        return derivation

    def learn_from_proof(
        self,
        premises: List[str],
        hypothesis: str,
        proof: List[Formula],
        target_proven: bool,
    ) -> float:
        premise_tokens = self._encode_formulas(premises)
        hypothesis_tokens = self._encode_formulas([hypothesis])

        scores = self.forward(premise_tokens, hypothesis_tokens)
        target = torch.tensor([1.0 if target_proven else 0.0], device=scores.device)

        loss = nn.functional.binary_cross_entropy(scores, target)
        loss.backward()
        return loss.item()


class NCoReProver(nn.Module):
    def __init__(
        self,
        embedding_dim: int = 256,
        hidden_dim: int = 512,
        num_relations: int = 10,
    ):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.num_relations = num_relations

        self.entity_embedding = nn.Embedding(10000, embedding_dim)
        self.relation_embedding = nn.Embedding(num_relations, embedding_dim)

        self.graph_encoder = nn.GraphConv(embedding_dim, hidden_dim)
        self.rule_learner = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_relations),
        )

        self.proof_network = nn.GRU(
            embedding_dim * 2, hidden_dim, num_layers=3, batch_first=True
        )

        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1),
            nn.Sigmoid(),
        )

    def forward(
        self,
        graph: Dict[str, Any],
        query: Tuple[str, str],
    ) -> torch.Tensor:
        entities = graph["entities"]
        relations = graph["relations"]
        edges = graph["edges"]

        entity_emb = self.entity_embedding(entities)
        relation_emb = self.relation_embedding(relations)

        query_emb = torch.cat(
            [
                self.entity_embedding(
                    torch.tensor([hash(query[0]) % 10000], device=entity_emb.device)
                ),
                self.relation_embedding(
                    torch.tensor(
                        [hash(query[1]) % self.num_relations],
                        device=relation_emb.device,
                    )
                ),
            ],
            dim=-1,
        )

        proof_hidden = torch.zeros(
            3, entity_emb.size(0), entity_emb.size(1), device=entity_emb.device
        )

        proof_steps = []
        for step in range(5):
            proof_input = query_emb.unsqueeze(1).expand(-1, entity_emb.size(1), -1)
            proof_out, proof_hidden = self.proof_network(proof_input, proof_hidden)
            proof_steps.append(proof_out[:, -1, :])

        proof_repr = torch.stack(proof_steps).mean(dim=0)
        score = self.classifier(proof_repr)
        return score

    def prove(
        self,
        knowledge_graph: Dict[str, Any],
        query: Tuple[str, str],
        max_steps: int = 20,
    ) -> Tuple[bool, float, List[Tuple[str, str, str]]]:
        with torch.no_grad():
            score = self.forward(knowledge_graph, query)
            proven = score.item() > 0.5

            proof_path = []
            if proven:
                proof_path = self._extract_proof_path(knowledge_graph, query)

            return proven, score.item(), proof_path

    def _extract_proof_path(
        self,
        knowledge_graph: Dict[str, Any],
        query: Tuple[str, str],
    ) -> List[Tuple[str, str, str]]:
        relations = knowledge_graph.get("relations", [])
        edges = knowledge_graph.get("edges", [])

        path = []
        for i in range(min(3, len(edges))):
            if i < len(edges):
                r = relations[i] if i < len(relations) else "related_to"
                path.append((query[0], r, f"intermediate_{i}"))
        path.append((query[0], query[1], query[1]))
        return path


class ATPSolver(nn.Module):
    def __init__(
        self,
        embedding_dim: int = 128,
        hidden_dim: int = 256,
    ):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim

        self.term_encoder = nn.LSTM(
            embedding_dim,
            hidden_dim,
            num_layers=2,
            batch_first=True,
            bidirectional=True,
        )

        self.unification_network = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

        self.resolution_network = nn.Sequential(
            nn.Linear(hidden_dim * 3, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

        self.strategy_learner = nn.Parameter(torch.randn(4))

    def forward(
        self,
        clauses: List[torch.Tensor],
        goal: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        clause_reprs = []
        for clause in clauses:
            output, _ = self.term_encoder(clause.unsqueeze(0))
            clause_repr = output.mean(dim=1)
            clause_reprs.append(clause_repr)

        clause_tensor = torch.stack(clause_reprs).squeeze(1)
        goal_output, _ = self.term_encoder(goal.unsqueeze(0))
        goal_repr = goal_output.mean(dim=1)

        unification_scores = []
        for i in range(len(clauses)):
            for j in range(i + 1, len(clauses)):
                pair = torch.cat([clause_tensor[i], clause_tensor[j]], dim=-1)
                score = self.unification_network(pair)
                unification_scores.append((i, j, score))

        resolution_scores = []
        for i in range(len(clauses)):
            triple = torch.cat(
                [clause_tensor[i], goal_repr, clause_tensor[i] * goal_repr], dim=-1
            )
            score = self.resolution_network(triple)
            resolution_scores.append((i, score))

        return {
            "unification": unification_scores,
            "resolution": resolution_scores,
            "strategy_weights": torch.softmax(self.strategy_learner, dim=0),
        }

    def solve(
        self,
        axioms: List[str],
        goal: str,
        timeout: int = 100,
    ) -> Tuple[bool, List[str]]:
        with torch.no_grad():
            axiom_tensors = [self._encode_term(ax) for ax in axioms]
            goal_tensor = self._encode_term(goal)

            result = self.forward(axiom_tensors, goal_tensor)

            resolution_scores = result["resolution"]
            if resolution_scores:
                best_idx = max(
                    range(len(resolution_scores)),
                    key=lambda i: resolution_scores[i][1].item(),
                )
                if resolution_scores[best_idx][1].item() > 0.5:
                    proof = [axioms[resolution_scores[best_idx][0]]]
                    return True, proof

            return False, []

    def _encode_term(self, term: str) -> torch.Tensor:
        chars = [ord(c) % 128 for c in term[:32]]
        while len(chars) < 32:
            chars.append(0)
        return torch.tensor(chars, dtype=torch.float32)
