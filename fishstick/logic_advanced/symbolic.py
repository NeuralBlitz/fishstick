import torch
import torch.nn as nn
from typing import List, Optional, Dict, Any, Tuple, Set
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


class SymbolicReasoner(nn.Module):
    def __init__(
        self,
        num_variables: int = 100,
        embedding_dim: int = 128,
        hidden_dim: int = 256,
    ):
        super().__init__()
        self.num_variables = num_variables
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim

        self.variable_embedding = nn.Embedding(num_variables, embedding_dim)

        self.encoder = nn.LSTM(
            embedding_dim, hidden_dim, num_layers=2, batch_first=True
        )

        self.logic_layer = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, len(LogicalConnective) + 1),
        )

        self.embedding_to_logic = nn.Linear(embedding_dim, hidden_dim)

        self.and_gate = nn.Parameter(torch.ones(1))
        self.or_gate = nn.Parameter(torch.zeros(1))

    def forward(
        self,
        formulas: torch.Tensor,
        context: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        batch_size, num_formulas, seq_len = formulas.shape

        embedded = self.variable_embedding(formulas)

        output, _ = self.encoder(embedded.view(batch_size * num_formulas, seq_len, -1))
        output = output.mean(dim=1)
        output = output.view(batch_size, num_formulas, -1)

        if context is not None:
            context_expanded = context.unsqueeze(1).expand(-1, num_formulas, -1)
            output = output + context_expanded

        logits = self.logic_layer(output)
        return logits

    def evaluate(
        self,
        formula: Formula,
        assignment: Dict[str, bool],
    ) -> TruthValue:
        if not formula.connective:
            atom = str(formula.atoms[0]) if formula.atoms else "false"
            val = assignment.get(atom, False)
            return TruthValue.TRUE if val else TruthValue.FALSE

        if formula.connective == LogicalConnective.NOT:
            child_val = self.evaluate(formula.atoms[0], assignment)
            if child_val == TruthValue.TRUE:
                return TruthValue.FALSE
            elif child_val == TruthValue.FALSE:
                return TruthValue.TRUE
            return TruthValue.UNKNOWN

        if formula.connective == LogicalConnective.AND:
            vals = [self.evaluate(f, assignment) for f in formula.atoms]
            if all(v == TruthValue.TRUE for v in vals):
                return TruthValue.TRUE
            if any(v == TruthValue.FALSE for v in vals):
                return TruthValue.FALSE
            return TruthValue.UNKNOWN

        if formula.connective == LogicalConnective.OR:
            vals = [self.evaluate(f, assignment) for f in formula.atoms]
            if any(v == TruthValue.TRUE for v in vals):
                return TruthValue.TRUE
            if all(v == TruthValue.FALSE for v in vals):
                return TruthValue.FALSE
            return TruthValue.UNKNOWN

        if formula.connective == LogicalConnective.IMPLIES:
            vals = [self.evaluate(f, assignment) for f in formula.atoms]
            if vals[0] == TruthValue.FALSE or vals[1] == TruthValue.TRUE:
                return TruthValue.TRUE
            if vals[0] == TruthValue.TRUE and vals[1] == TruthValue.FALSE:
                return TruthValue.FALSE
            return TruthValue.UNKNOWN

        return TruthValue.UNKNOWN

    def entail(
        self,
        premises: List[Formula],
        conclusion: Formula,
    ) -> bool:
        all_vars = self._collect_variables(premises + [conclusion])
        all_assignments = self._generate_assignments(all_vars)

        for assignment in all_assignments:
            all_true = all(
                self.evaluate(p, assignment) == TruthValue.TRUE for p in premises
            )
            if all_true:
                if self.evaluate(conclusion, assignment) != TruthValue.TRUE:
                    return False
        return True

    def _collect_variables(self, formulas: List[Formula]) -> Set[str]:
        variables = set()
        for formula in formulas:
            if not formula.connective:
                if formula.atoms:
                    variables.add(str(formula.atoms[0]))
            else:
                variables.update(self._collect_variables(formula.atoms))
        return variables

    def _generate_assignments(self, variables: Set[str]) -> List[Dict[str, bool]]:
        var_list = sorted(variables)
        num_assignments = 2 ** len(var_list)
        assignments = []
        for i in range(num_assignments):
            assignment = {}
            for j, var in enumerate(var_list):
                assignment[var] = bool((i >> j) & 1)
            assignments.append(assignment)
        return assignments


class LogicalNeuralNetwork(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 256,
        output_dim: int = 10,
        num_layers: int = 3,
        fuzzy: bool = True,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.fuzzy = fuzzy

        self.input_projection = nn.Linear(input_dim, hidden_dim)

        self.logic_layers = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(hidden_dim, hidden_dim),
                    nn.ReLU(),
                    nn.Linear(hidden_dim, hidden_dim),
                )
                for _ in range(num_layers)
            ]
        )

        self.gates = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(hidden_dim, hidden_dim),
                    nn.Sigmoid(),
                )
                for _ in range(num_layers)
            ]
        )

        self.output_layer = nn.Linear(hidden_dim, output_dim)

        if fuzzy:
            self.t_norm = self._godel_t_norm
        else:
            self.t_norm = self._classical_and

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.input_projection(x)

        for i, (logic_layer, gate) in enumerate(zip(self.logic_layers, self.gates)):
            h = logic_layer(x)
            g = gate(x)

            if self.fuzzy:
                x = g * h + (1 - g) * self.t_norm(x, h)
            else:
                x = torch.min(x, h) if i % 2 == 0 else torch.max(x, h)

        return self.output_layer(x)

    def _godel_t_norm(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        return torch.min(a, b)

    def _classical_and(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        return a * b

    def explain_prediction(
        self,
        x: torch.Tensor,
    ) -> Dict[str, Any]:
        x = self.input_projection(x)

        explanations = []
        for i, (logic_layer, gate) in enumerate(zip(self.logic_layers, self.gates)):
            h = logic_layer(x)
            g = gate(x)
            explanations.append(
                {
                    "layer": i,
                    "gate_activation": g.mean().item(),
                    "hidden_norm": h.norm().item(),
                }
            )

        return {
            "layer_activations": explanations,
            "input_importance": x.abs().mean(dim=-1).tolist(),
        }


class LNN(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        embedding_dim: int = 128,
        hidden_dim: int = 256,
        num_rules: int = 10,
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim

        self.embedding = nn.Embedding(vocab_size, embedding_dim)

        self.atom_encoder = nn.LSTM(
            embedding_dim, hidden_dim, num_layers=2, batch_first=True
        )

        self.rule_encoder = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_rules),
        )

        self.wm_update = nn.GRUCell(hidden_dim * 2, hidden_dim)

        self.truth_predictor = nn.Linear(hidden_dim, 1)

        self.K = nn.Parameter(torch.randn(hidden_dim, hidden_dim))

    def forward(
        self,
        atoms: torch.Tensor,
        rules: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        batch_size = atoms.size(0)

        embedded = self.embedding(atoms)
        encoded, _ = self.atom_encoder(embedded)
        atom_repr = encoded.mean(dim=1)

        if rules is not None:
            rule_embedded = self.embedding(rules)
            rule_encoded, _ = self.atom_encoder(rule_embedded)
            rule_repr = rule_encoded.mean(dim=1)

            rule_logits = self.rule_encoder(torch.cat([atom_repr, rule_repr], dim=-1))

            for i in range(rule_logits.size(1)):
                rule_strength = torch.softmax(rule_logits, dim=1)[:, i : i + 1]
                weighted_rule = rule_strength * rule_repr
                atom_repr = self.wm_update(
                    torch.cat([atom_repr, weighted_rule], dim=-1), atom_repr
                )

        truth_values = self.truth_predictor(atom_repr)
        truth_values = torch.sigmoid(truth_values)

        return {
            "truth_values": truth_values,
            "atom_repr": atom_repr,
            "rule_logits": rule_logits if rules is not None else None,
        }

    def learn_rules(
        self,
        examples: List[Tuple[List[str], str]],
    ) -> List[str]:
        learned_rules = []
        return learned_rules

    def evaluate_formula(
        self,
        formula: str,
        knowledge_base: Dict[str, float],
    ) -> float:
        parts = formula.split()
        if len(parts) == 1:
            return knowledge_base.get(parts[0], 0.5)

        if "∧" in formula or "and" in formula:
            tokens = formula.replace("∧", " and ").split(" and ")
            return min(self.evaluate_formula(t, knowledge_base) for t in tokens)

        if "∨" in formula or "or" in formula:
            tokens = formula.replace("∨", " or ").split(" or ")
            return max(self.evaluate_formula(t, knowledge_base) for t in tokens)

        if "→" in formula or "implies" in formula:
            tokens = formula.replace("→", " implies ").split(" implies ")
            antecedent = self.evaluate_formula(tokens[0], knowledge_base)
            consequent = self.evaluate_formula(tokens[1], knowledge_base)
            return min(1.0 - antecedent + consequent, 1.0)

        return 0.5

    def forward_reasoning(
        self,
        initial_facts: Dict[str, float],
        rules: List[Tuple[str, str, str]],
        iterations: int = 10,
    ) -> Dict[str, float]:
        knowledge = initial_facts.copy()

        for _ in range(iterations):
            for antecedent, consequent, connective in rules:
                ant_val = self.evaluate_formula(antecedent, knowledge)
                con_val = knowledge.get(consequent, 0.0)

                if connective == "implies":
                    new_val = min(1.0 - ant_val + con_val, 1.0)
                elif connective == "and":
                    new_val = min(ant_val, con_val)
                elif connective == "or":
                    new_val = max(ant_val, con_val)
                else:
                    new_val = con_val

                knowledge[consequent] = max(knowledge.get(consequent, 0.0), new_val)

        return knowledge
