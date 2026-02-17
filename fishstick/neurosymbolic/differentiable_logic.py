"""
Differentiable Logic Layer Implementations.

Implements:
- Soft logic gates (AND, OR, NOT) with continuous relaxations
- Logic Tensor Networks (LTN)
- Differentiable SAT solvers
- Logical regularization for neural networks
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Dict, Tuple, Union
import torch
from torch import Tensor, nn
import torch.nn.functional as F
import numpy as np


class GumbelSoftmax(torch.nn.Module):
    """Gumbel-Softmax relaxation for discrete distributions.

    Provides a differentiable approximation to categorical sampling.
    """

    def __init__(self, temperature: float = 1.0, hard: bool = False):
        """Initialize Gumbel-Softmax.

        Args:
            temperature: Temperature for softmax (lower = more discrete)
            hard: If True, use straight-through estimator
        """
        super().__init__()
        self.temperature = temperature
        self.hard = hard

    def forward(self, logits: Tensor) -> Tensor:
        """Sample from Gumbel-Softmax distribution.

        Args:
            logits: Unnormalized log probabilities [batch, n_classes]

        Returns:
            Soft samples [batch, n_classes]
        """
        gumbels = -torch.empty_like(logits).exponential_().log()
        gumbels = (logits + gumbels) / self.temperature
        y_soft = F.softmax(gumbels, dim=-1)

        if self.hard:
            index = y_soft.max(dim=-1, keepdim=True)[1]
            y_hard = torch.zeros_like(logits).scatter_(-1, index, 1.0)
            return (y_hard - y_soft).detach() + y_soft
        return y_soft


class SoftLogicGate(nn.Module):
    """Differentiable soft logic gates with learnable semantics.

    Implements continuous relaxations of Boolean logic:
    - AND: minimum/conjunction
    - OR: maximum/disjunction
    - NOT: 1 - x
    """

    def __init__(self, gate_type: str = "AND", learnable: bool = True):
        """Initialize soft logic gate.

        Args:
            gate_type: Type of gate ("AND", "OR", "NOT", "XOR")
            learnable: Whether to use learnable soft operations
        """
        super().__init__()
        self.gate_type = gate_type.upper()

        if learnable:
            if self.gate_type == "AND":
                self.weight = nn.Parameter(torch.ones(1))
            elif self.gate_type == "OR":
                self.weight = nn.Parameter(torch.ones(1))
            else:
                self.weight = nn.Parameter(torch.ones(1))
        else:
            self.register_parameter("weight", None)

    def forward(self, *inputs: Tensor) -> Tensor:
        """Apply soft logic gate.

        Args:
            inputs: Input tensors (1 or 2 depending on gate type)

        Returns:
            Soft logical output
        """
        if len(inputs) == 1:
            x = inputs[0]
            if x.dim() > 1:
                pass
            elif x.numel() > 1:
                x = x.unsqueeze(-1)
        elif len(inputs) == 2:
            x = torch.stack(inputs, dim=-1)
        else:
            x = torch.stack(inputs, dim=-1)

        if self.gate_type == "NOT":
            return 1.0 - x.squeeze(-1) if x.dim() > 1 else 1.0 - x

        if self.gate_type == "AND":
            if self.weight is not None:
                weights = torch.sigmoid(self.weight)
                return self._soft_and_weighted(x, weights)
            return self._soft_and(x)

        if self.gate_type == "OR":
            if self.weight is not None:
                weights = torch.sigmoid(self.weight)
                return self._soft_or_weighted(x, weights)
            return self._soft_or(x)

        if self.gate_type == "XOR":
            if len(inputs) >= 2:
                return self._soft_xor(inputs[0], inputs[1])
            return x.squeeze(-1)

        raise ValueError(f"Unknown gate type: {self.gate_type}")

    def _soft_and(self, inputs: Tensor) -> Tensor:
        """Soft AND using product."""
        if inputs.dim() == 1:
            return inputs
        return inputs.prod(dim=-1)

    def _soft_and_weighted(self, inputs: Tensor, weight: Tensor) -> Tensor:
        """Weighted soft AND."""
        if inputs.dim() == 1:
            return inputs
        return (inputs**weight).prod(dim=-1)

    def _soft_or(self, inputs: Tensor) -> Tensor:
        """Soft OR using 1 - product(1 - x)."""
        if inputs.dim() == 1:
            return inputs
        return 1.0 - ((1.0 - inputs).prod(dim=-1))

    def _soft_or_weighted(self, inputs: Tensor, weight: Tensor) -> Tensor:
        """Weighted soft OR."""
        if inputs.dim() == 1:
            return inputs
        return 1.0 - (((1.0 - inputs) ** weight).prod(dim=-1))

    def _soft_xor(self, a: Tensor, b: Tensor) -> Tensor:
        """Soft XOR using squared difference."""
        return (a - b) ** 2


class DifferentiableClause(nn.Module):
    """Differentiable SAT clause (disjunction of literals).

    Implements a soft version of: (l1 OR l2 OR ... OR ln)
    """

    def __init__(
        self,
        n_literals: int,
        t_norm: str = "goedel",
        learnable_weights: bool = True,
    ):
        """Initialize differentiable clause.

        Args:
            n_literals: Number of literals in the clause
            t_norm: T-norm to use ("goedel", "product", "lukasiewicz")
            learnable_weights: Whether clause has learnable weights
        """
        super().__init__()
        self.n_literals = n_literals
        self.t_norm = t_norm

        if learnable_weights:
            self.weights = nn.Parameter(torch.ones(n_literals))
        else:
            self.register_parameter("weights", None)

    def forward(self, literals: Tensor) -> Tensor:
        """Evaluate clause satisfaction.

        Args:
            literals: Truth values of literals [batch, n_literals]

        Returns:
            Clause satisfaction degree [batch]
        """
        if self.weights is not None:
            weights = F.softmax(self.weights, dim=-1)
            weighted_literals = literals * weights
        else:
            weighted_literals = literals

        if self.t_norm == "goedel":
            return weighted_literals.max(dim=-1)[0]
        elif self.t_norm == "product":
            return weighted_literals.prod(dim=-1)
        elif self.t_norm == "lukasiewicz":
            return torch.clamp(weighted_literals.sum(dim=-1), 0, 1)
        raise ValueError(f"Unknown t_norm: {self.t_norm}")


class DifferentiableFormula(nn.Module):
    """Differentiable first-order logic formula.

    Supports:
    - Universal and existential quantifiers
    - Implication
    - Equivalence
    - Various t-norms
    """

    def __init__(
        self,
        formula_type: str = "clause",
        n_vars: int = 2,
        t_norm: str = "goedel",
    ):
        """Initialize differentiable formula.

        Args:
            formula_type: Type of formula ("clause", "horn", "general")
            n_vars: Number of variables
            t_norm: T-norm for soft logic
        """
        super().__init__()
        self.formula_type = formula_type
        self.n_vars = n_vars
        self.t_norm = t_norm

        if formula_type == "horn":
            self.clauses = nn.ModuleList(
                [DifferentiableClause(n_vars, t_norm) for _ in range(n_vars)]
            )

    def forward(
        self,
        predicates: Tensor,
        quantifier_values: Optional[Tensor] = None,
    ) -> Tensor:
        """Evaluate formula.

        Args:
            predicates: Predicate truth values [batch, n_predicates]
            quantifier_values: Values for quantified variables

        Returns:
            Formula satisfaction degree
        """
        if self.formula_type == "clause":
            clause = DifferentiableClause(predicates.size(-1), self.t_norm)
            return clause(predicates)

        elif self.formula_type == "horn":
            clause_satisfactions = []
            for clause in self.clauses:
                sat = clause(predicates)
                clause_satisfactions.append(sat)
            return torch.stack(clause_satisfactions).min(dim=0)[0]

        elif self.formula_type == "general":
            return predicates.mean(dim=-1)

        raise ValueError(f"Unknown formula type: {self.formula_type}")

    def imply(self, antecedent: Tensor, consequent: Tensor) -> Tensor:
        """Material implication: A -> B = ¬A OR B.

        Args:
            antecedent: Truth value of antecedent
            consequent: Truth value of consequent

        Returns:
            Implication truth value
        """
        not_antecedent = 1.0 - antecedent
        if self.t_norm == "goedel":
            return torch.max(not_antecedent, consequent)
        elif self.t_norm == "product":
            return not_antecedent * consequent
        elif self.t_norm == "lukasiewicz":
            return torch.clamp(not_antecedent + consequent, 0, 1)
        raise ValueError(f"Unknown t_norm: {self.t_norm}")

    def equivalence(self, a: Tensor, b: Tensor) -> Tensor:
        """Logical equivalence: A <-> B.

        Args:
            a: First truth value
            b: Second truth value

        Returns:
            Equivalence truth value
        """
        impl_ab = self.imply(a, b)
        impl_ba = self.imply(b, a)
        return torch.min(impl_ab, impl_ba)

    def forall(self, values: Tensor, dim: int = -1) -> Tensor:
        """Universal quantifier: FOR ALL x. P(x)

        Args:
            values: Truth values across domain
            dim: Dimension to quantify over

        Returns:
            Quantified truth value
        """
        if self.t_norm == "goedel":
            return values.min(dim=dim)[0]
        elif self.t_norm == "product":
            return values.prod(dim=dim)
        elif self.t_norm == "lukasiewicz":
            n = values.size(dim)
            return torch.clamp(values.sum(dim=dim) - (n - 1), 0, 1)
        raise ValueError(f"Unknown t_norm: {self.t_norm}")

    def exists(self, values: Tensor, dim: int = -1) -> Tensor:
        """Existential quantifier: EXISTS x. P(x)

        Args:
            values: Truth values across domain
            dim: Dimension to quantify over

        Returns:
            Quantified truth value
        """
        if self.t_norm == "goedel":
            return values.max(dim=dim)[0]
        elif self.t_norm == "product":
            return 1.0 - ((1.0 - values).prod(dim=dim))
        elif self.t_norm == "lukasiewicz":
            return torch.clamp(values.sum(dim=dim), 0, 1)
        raise ValueError(f"Unknown t_norm: {self.t_norm}")


class LogicTensorNetwork(nn.Module):
    """Logic Tensor Network (LTN) implementation.

    Combines neural networks with fuzzy logic for relational reasoning.
    From: "Logic Tensor Networks: Deep Learning and Logical Reasoning"
    """

    def __init__(
        self,
        n_predicates: int,
        n_variables: int,
        embedding_dim: int = 64,
        t_norm: str = "goedel",
    ):
        """Initialize LTN.

        Args:
            n_predicates: Number of predicate functions
            n_variables: Number of variables in domain
            embedding_dim: Dimension of entity embeddings
            t_norm: T-norm for soft logic
        """
        super().__init__()
        self.n_predicates = n_predicates
        self.n_variables = n_variables
        self.embedding_dim = embedding_dim
        self.t_norm = t_norm

        self.entity_embeddings = nn.Embedding(n_variables, embedding_dim)

        self.predicates = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(embedding_dim, 64),
                    nn.ReLU(),
                    nn.Linear(64, 32),
                    nn.ReLU(),
                    nn.Linear(32, 1),
                    nn.Sigmoid(),
                )
                for _ in range(n_predicates)
            ]
        )

        self.formula = DifferentiableFormula(
            formula_type="general", n_vars=n_variables, t_norm=t_norm
        )

    def forward(
        self,
        entity_ids: Tensor,
        predicate_idx: int,
    ) -> Tensor:
        """Evaluate predicate on entities.

        Args:
            entity_ids: Entity indices [batch, arity]
            predicate_idx: Index of predicate to evaluate

        Returns:
            Truth values [batch]
        """
        embeddings = self.entity_embeddings(entity_ids)
        return self.predicates[predicate_idx](embeddings).squeeze(-1)

    def evaluate_formula(
        self,
        predicate_results: Tensor,
    ) -> Tensor:
        """Evaluate a formula on predicate results.

        Args:
            predicate_results: Results from predicate evaluations

        Returns:
            Formula satisfaction degree
        """
        return self.formula(predicate_results)

    def sat_loss(self, formula_results: Tensor) -> Tensor:
        """Compute satisfaction loss (to maximize).

        Args:
            formula_results: Results of formula evaluations

        Returns:
            Negative satisfaction (loss to minimize)
        """
        return -formula_results.mean()


class DifferentiableSATSolver(nn.Module):
    """Differentiable SAT solver using gradient-based approach.

    Learns truth assignments that satisfy a CNF formula.
    """

    def __init__(
        self,
        n_variables: int,
        n_clauses: int,
        clause_size: int = 3,
    ):
        """Initialize SAT solver.

        Args:
            n_variables: Number of Boolean variables
            n_clauses: Number of clauses in CNF
            clause_size: Size of each clause (k-CNF)
        """
        super().__init__()
        self.n_variables = n_variables
        self.n_clauses = n_clauses
        self.clause_size = clause_size

        self.assignment_logits = nn.Parameter(torch.zeros(n_variables))

        clause_template = []
        for i in range(n_clauses):
            vars_in_clause = torch.randint(0, n_variables, (clause_size,))
            signs = torch.randint(0, 2, (clause_size,))
            clause_template.append((vars_in_clause, signs))
        self.clause_template = clause_template

    def get_assignments(self) -> Tensor:
        """Get current truth assignments.

        Returns:
            Truth assignments [n_variables] (continuous in [0, 1])
        """
        return torch.sigmoid(self.assignment_logits)

    def forward(self) -> Dict[str, Tensor]:
        """Solve SAT problem.

        Returns:
            Dictionary with satisfaction scores
        """
        assignments = self.get_assignments()

        clause_satisfactions = []
        for vars_in_clause, signs in self.clause_template:
            literal_values = []
            for var_idx, sign in zip(vars_in_clause.tolist(), signs.tolist()):
                lit_value = (
                    assignments[var_idx] if sign == 1 else (1.0 - assignments[var_idx])
                )
                literal_values.append(lit_value)

            clause_sat = torch.stack(literal_values).max()
            clause_satisfactions.append(clause_sat)

        all_satisfied = torch.stack(clause_satisfactions).min()

        return {
            "assignments": assignments,
            "clause_satisfactions": torch.stack(clause_satisfactions),
            "all_satisfied": all_satisfied,
            "n_satisfied": (torch.stack(clause_satisfactions) > 0.5).float().sum(),
        }

    def sat_loss(self) -> Tensor:
        """Compute SAT loss (to minimize).

        Returns:
            Loss value (0 = fully satisfied)
        """
        result = self.forward()
        return 1.0 - result["all_satisfied"]


class LogicalRegularizer(nn.Module):
    """Regularizer for enforcing logical constraints in neural networks.

    Encourages network outputs to satisfy given logical formulas.
    """

    def __init__(
        self,
        formula: DifferentiableFormula,
        weight: float = 1.0,
    ):
        """Initialize logical regularizer.

        Args:
            formula: Logical formula to enforce
            weight: Weight of regularization term
        """
        super().__init__()
        self.formula = formula
        self.weight = weight

    def forward(self, outputs: Tensor) -> Tensor:
        """Compute regularization loss.

        Args:
            outputs: Network outputs to regularize

        Returns:
            Regularization loss
        """
        formula_result = self.formula(outputs)
        return self.weight * (1.0 - formula_result.mean())


class NeuralPredicate(nn.Module):
    """Neural network that represents a logical predicate.

    Can be trained to approximate arbitrary Boolean relations.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 64,
        n_layers: int = 2,
        output_type: str = "sigmoid",
    ):
        """Initialize neural predicate.

        Args:
            input_dim: Input feature dimension
            hidden_dim: Hidden layer dimension
            n_layers: Number of hidden layers
            output_type: Output activation ("sigmoid", "hard", "gumbel")
        """
        super().__init__()
        self.input_dim = input_dim
        self.output_type = output_type

        layers = []
        dims = [input_dim] + [hidden_dim] * n_layers + [1]
        for i in range(len(dims) - 1):
            layers.append(nn.Linear(dims[i], dims[i + 1]))
            if i < len(dims) - 2:
                layers.append(nn.ReLU())
        self.network = nn.Sequential(*layers)

    def forward(self, x: Tensor) -> Tensor:
        """Evaluate predicate on inputs.

        Args:
            x: Input features [batch, input_dim]

        Returns:
            Truth value [batch]
        """
        logits = self.network(x)
        if self.output_type == "sigmoid":
            return torch.sigmoid(logits)
        elif self.output_type == "hard":
            return (logits > 0).float()
        elif self.output_type == "gumbel":
            gs = GumbelSoftmax(temperature=0.1)
            return gs(logits).squeeze(-1)
        return logits.squeeze(-1)
