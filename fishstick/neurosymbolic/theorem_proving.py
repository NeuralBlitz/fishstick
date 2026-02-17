"""
Neural Theorem Proving Basics.

Implements:
- Proof tree representations
- Neural unification
- Backward chaining with neural guidance
- Natural deduction system
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Dict, Set, Tuple, Callable, Any
from enum import Enum, auto
import torch
from torch import Tensor, nn
import torch.nn.functional as F
import numpy as np
from collections import deque


class FormulaType(Enum):
    """Types of logical formulas."""

    ATOM = auto()
    NEGATION = auto()
    CONJUNCTION = auto()
    DISJUNCTION = auto()
    IMPLICATION = auto()
    UNIVERSAL = auto()
    EXISTENTIAL = auto()


@dataclass
class Formula:
    """Logical formula representation.

    Attributes:
        formula_type: Type of formula
        predicate: Predicate name (for atoms)
        args: Arguments (for atoms)
        subformulas: Subformulas (for compound formulas)
        variable: Bound variable (for quantified formulas)
    """

    formula_type: FormulaType
    predicate: Optional[str] = None
    args: Tuple[Any, ...] = field(default_factory=tuple)
    subformulas: Tuple["Formula, ..."] = field(default_factory=tuple)
    variable: Optional[str] = None

    def __str__(self) -> str:
        """String representation."""
        if self.formula_type == FormulaType.ATOM:
            if self.args:
                return f"{self.predicate}({', '.join(str(a) for a in self.args)})"
            return str(self.predicate)

        elif self.formula_type == FormulaType.NEGATION:
            return f"¬{self.subformulas[0]}"

        elif self.formula_type == FormulaType.CONJUNCTION:
            return f"({self.subformulas[0]} ∧ {self.subformulas[1]})"

        elif self.formula_type == FormulaType.DISJUNCTION:
            return f"({self.subformulas[0]} ∨ {self.subformulas[1]})"

        elif self.formula_type == FormulaType.IMPLICATION:
            return f"({self.subformulas[0]} → {self.subformulas[1]})"

        elif self.formula_type == FormulaType.UNIVERSAL:
            return f"∀{self.variable}. {self.subformulas[0]}"

        elif self.formula_type == FormulaType.EXISTENTIAL:
            return f"∃{self.variable}. {self.subformulas[0]}"

        return "<formula>"

    def get_atoms(self) -> Set[Tuple[str, Tuple]]:
        """Get all atoms in the formula."""
        if self.formula_type == FormulaType.ATOM:
            return {(self.predicate, self.args)}

        atoms = set()
        for sub in self.subformulas:
            atoms.update(sub.get_atoms())
        return atoms


@dataclass
class Substitution:
    """Variable substitution mapping.

    Attributes:
        mapping: Dictionary from variables to terms
    """

    mapping: Dict[str, Any] = field(default_factory=dict)

    def __add__(self, other: "Substitution") -> "Substitution":
        """Compose substitutions."""
        new_mapping = self.mapping.copy()
        new_mapping.update(other.mapping)
        return Substitution(new_mapping)

    def apply(self, term: Any) -> Any:
        """Apply substitution to a term."""
        if isinstance(term, str) and term in self.mapping:
            return self.mapping[term]
        if isinstance(term, tuple):
            return tuple(self.apply(t) for t in term)
        return term

    def compose(self, other: "Substitution") -> "Substitution":
        """Compose with another substitution."""
        composed = {}
        for var, term in self.mapping.items():
            composed[var] = other.apply(term)
        for var, term in other.mapping.items():
            if var not in self.mapping:
                composed[var] = term
        return Substitution(composed)


class Unifier:
    """Unification algorithm for first-order logic.

    Computes most general unifiers (MGUs).
    """

    @staticmethod
    def unify(
        term1: Any, term2: Any, subst: Optional[Substitution] = None
    ) -> Optional[Substitution]:
        """Unify two terms.

        Args:
            term1: First term
            term2: Second term
            subst: Current substitution

        Returns:
            MGU or None if unification fails
        """
        if subst is None:
            subst = Substitution()

        term1 = subst.apply(term1)
        term2 = subst.apply(term2)

        if term1 == term2:
            return subst

        if isinstance(term1, str) and term1.islower() and term1[0].isalpha():
            return Unifier._unify_variable(term1, term2, subst)

        if isinstance(term2, str) and term2.islower() and term2[0].isalpha():
            return Unifier._unify_variable(term2, term1, subst)

        if isinstance(term1, tuple) and isinstance(term2, tuple):
            if len(term1) != len(term2):
                return None
            for t1, t2 in zip(term1, term2):
                subst = Unifier.unify(t1, t2, subst)
                if subst is None:
                    return None
            return subst

        return None

    @staticmethod
    def _unify_variable(
        var: str, term: Any, subst: Substitution
    ) -> Optional[Substitution]:
        """Unify a variable with a term."""
        if var in subst.mapping:
            return Unifier.unify(subst.mapping[var], term, subst)

        if Unifier._occurs_check(var, term):
            return None

        new_mapping = subst.mapping.copy()
        new_mapping[var] = term
        return Substitution(new_mapping)

    @staticmethod
    def _occurs_check(var: str, term: Any) -> bool:
        """Check if variable occurs in term."""
        if var == term:
            return True
        if isinstance(term, tuple):
            return any(Unifier._occurs_check(var, t) for t in term)
        return False


class NeuralUnifier(nn.Module):
    """Neural network for learning unification.

    Learns to predict unification success and substitutions.
    """

    def __init__(
        self,
        vocab_size: int = 100,
        embed_dim: int = 64,
    ):
        """Initialize neural unifier.

        Args:
            vocab_size: Vocabulary size
            embed_dim: Embedding dimension
        """
        super().__init__()
        self.vocab_size = vocab_size
        self.embed_dim = embed_dim

        self.embedding = nn.Embedding(vocab_size, embed_dim)

        self.encoder = nn.LSTM(
            embed_dim,
            embed_dim,
            batch_first=True,
        )

        self.unify_predictor = nn.Sequential(
            nn.Linear(embed_dim * 2, embed_dim),
            nn.ReLU(),
            nn.Linear(embed_dim, 1),
            nn.Sigmoid(),
        )

        self.subst_predictor = nn.Sequential(
            nn.Linear(embed_dim * 2, embed_dim),
            nn.ReLU(),
            nn.Linear(embed_dim, vocab_size),
        )

    def forward(
        self,
        term1_ids: Tensor,
        term2_ids: Tensor,
    ) -> Dict[str, Tensor]:
        """Predict unification.

        Args:
            term1_ids: First term IDs [batch, seq_len]
            term2_ids: Second term IDs [batch, seq_len]

        Returns:
            Dictionary with predictions
        """
        embed1 = self.embedding(term1_ids)
        embed2 = self.embedding(term2_ids)

        _, (h1, _) = self.encoder(embed1)
        _, (h2, _) = self.encoder(embed2)

        h1 = h1.squeeze(0)
        h2 = h2.squeeze(0)

        combined = torch.cat([h1, h2], dim=-1)

        unify_prob = self.unify_predictor(combined).squeeze(-1)

        subst_logits = self.subst_predictor(combined)

        return {
            "unify_prob": unify_prob,
            "subst_logits": subst_logits,
        }


@dataclass
class ProofStep:
    """Step in a proof.

    Attributes:
        rule: Inference rule used
        premises: Premises used
        conclusion: Conclusion drawn
        substitution: Substitution applied
    """

    rule: str
    premises: List[int]
    conclusion: Formula
    substitution: Optional[Substitution] = None


@dataclass
class ProofTree:
    """Proof tree representation.

    Attributes:
        goal: Goal formula
        proof_steps: List of proof steps
        is_proven: Whether the proof succeeded
    """

    goal: Formula
    proof_steps: List[ProofStep] = field(default_factory=list)
    is_proven: bool = False

    def add_step(self, step: ProofStep) -> None:
        """Add a proof step."""
        self.proof_steps.append(step)

    def __str__(self) -> str:
        """String representation."""
        lines = [f"Goal: {self.goal}"]
        if self.is_proven:
            lines.append("Proof:")
            for i, step in enumerate(self.proof_steps):
                lines.append(f"  {i + 1}. {step.rule}: {step.conclusion}")
        else:
            lines.append("Proof failed")
        return "\n".join(lines)


class NaturalDeduction:
    """Natural deduction proof system.

    Implements common inference rules.
    """

    def __init__(self):
        """Initialize natural deduction system."""
        self.rules = {
            "and_intro": self._and_intro,
            "and_elim_l": self._and_elim_l,
            "and_elim_r": self._and_elim_r,
            "or_intro_l": self._or_intro_l,
            "or_intro_r": self._or_intro_r,
            "or_elim": self._or_elim,
            "imply_intro": self._imply_intro,
            "imply_elim": self._imply_elim,
            "not_intro": self._not_intro,
            "not_elim": self._not_elim,
            "forall_intro": self._forall_intro,
            "forall_elim": self._forall_elim,
            "exists_intro": self._exists_intro,
            "exists_elim": self._exists_elim,
        }

    def _and_intro(self, f1: Formula, f2: Formula) -> Formula:
        """Conjunction introduction: A, B → A ∧ B."""
        return Formula(
            formula_type=FormulaType.CONJUNCTION,
            subformulas=(f1, f2),
        )

    def _and_elim_l(self, f: Formula) -> Formula:
        """Conjunction elimination (left): A ∧ B → A."""
        return f.subformulas[0]

    def _and_elim_r(self, f: Formula) -> Formula:
        """Conjunction elimination (right): A ∧ B → B."""
        return f.subformulas[1]

    def _or_intro_l(self, f1: Formula, f2: Formula) -> Formula:
        """Disjunction introduction (left): A → A ∨ B."""
        return Formula(
            formula_type=FormulaType.DISJUNCTION,
            subformulas=(f1, f2),
        )

    def _or_intro_r(self, f1: Formula, f2: Formula) -> Formula:
        """Disjunction introduction (right): B → A ∨ B."""
        return Formula(
            formula_type=FormulaType.DISJUNCTION,
            subformulas=(f1, f2),
        )

    def _or_elim(
        self,
        f_or: Formula,
        f1: Formula,
        f2: Formula,
    ) -> Formula:
        """Disjunction elimination: A ∨ B, A → C, B → C → C."""
        return f1

    def _imply_intro(self, premise: Formula, conclusion: Formula) -> Formula:
        """Implication introduction: (assume A) → B → (A → B)."""
        return Formula(
            formula_type=FormulaType.IMPLICATION,
            subformulas=(premise, conclusion),
        )

    def _imply_elim(self, f_imp: Formula, f_antecedent: Formula) -> Formula:
        """Implication elimination (modus ponens): A → B, A → B."""
        return f_imp.subformulas[1]

    def _not_intro(self, assumption: Formula, contradiction: Formula) -> Formula:
        """Negation introduction: assume A, derive false → ¬A."""
        return Formula(
            formula_type=FormulaType.NEGATION,
            subformulas=(assumption,),
        )

    def _not_elim(self, f_not: Formula, f: Formula) -> Formula:
        """Negation elimination: ¬A, A → false."""
        return Formula(
            formula_type=FormulaType.ATOM,
            predicate="false",
        )

    def _forall_intro(self, var: str, f: Formula) -> Formula:
        """Universal introduction."""
        return Formula(
            formula_type=FormulaType.UNIVERSAL,
            variable=var,
            subformulas=(f,),
        )

    def _forall_elim(self, f: Formula, term: Any) -> Formula:
        """Universal elimination."""
        return f.subformulas[0]

    def _exists_intro(self, var: str, f: Formula) -> Formula:
        """Existential introduction."""
        return Formula(
            formula_type=FormulaType.EXISTENTIAL,
            variable=var,
            subformulas=(f,),
        )

    def _exists_elim(self, f: Formula, var: str, conclusion: Formula) -> Formula:
        """Existential elimination."""
        return conclusion


class NeuralTheoremProver(nn.Module):
    """Neural theorem prover with learned guidance.

    Uses neural networks to guide proof search.
    """

    def __init__(
        self,
        n_atoms: int = 100,
        embed_dim: int = 64,
        n_hidden: int = 128,
    ):
        """Initialize neural theorem prover.

        Args:
            n_atoms: Number of atoms
            embed_dim: Embedding dimension
            n_hidden: Hidden dimension
        """
        super().__init__()
        self.n_atoms = n_atoms
        self.embed_dim = embed_dim

        self.atom_embedding = nn.Embedding(n_atoms, embed_dim)

        self.encoder = nn.LSTM(
            embed_dim,
            embed_dim,
            batch_first=True,
        )

        self.rule_predictor = nn.Sequential(
            nn.Linear(embed_dim * 2, n_hidden),
            nn.ReLU(),
            nn.Linear(n_hidden, len(NaturalDeduction().rules)),
        )

        self.goal_scorer = nn.Sequential(
            nn.Linear(embed_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
        )

    def forward(
        self,
        goal_ids: Tensor,
        premise_ids: Tensor,
    ) -> Dict[str, Tensor]:
        """Predict proof steps.

        Args:
            goal_ids: Goal atom IDs [batch, seq_len]
            premise_ids: Premise atom IDs [batch, seq_len]

        Returns:
            Predictions
        """
        goal_embed = self.atom_embedding(goal_ids)
        premise_embed = self.atom_embedding(premise_ids)

        _, (goal_h, _) = self.encoder(goal_embed)
        _, (premise_h, _) = self.encoder(premise_embed)

        goal_h = goal_h.squeeze(0)
        premise_h = premise_h.squeeze(0)

        combined = torch.cat([goal_h, premise_h], dim=-1)

        rule_scores = self.rule_predictor(combined)
        goal_scores = self.goal_scorer(goal_h)

        return {
            "rule_scores": rule_scores,
            "goal_scores": goal_scores,
        }

    def select_rule(self, rule_scores: Tensor) -> str:
        """Select inference rule based on scores."""
        rule_names = list(NaturalDeduction().rules.keys())
        idx = rule_scores.argmax().item()
        return rule_names[idx]


class BackwardChainer:
    """Backward chaining theorem prover.

    Performs goal-directed proof search.
    """

    def __init__(
        self,
        knowledge_base: Dict[Formula, List[Formula]],
    ):
        """Initialize backward chainer.

        Args:
            knowledge_base: Mapping from conclusions to premises
        """
        self.knowledge_base = knowledge_base

    def prove(
        self,
        goal: Formula,
        depth: int = 0,
        max_depth: int = 10,
    ) -> Optional[ProofTree]:
        """Attempt to prove goal using backward chaining.

        Args:
            goal: Goal formula
            depth: Current search depth
            max_depth: Maximum search depth

        Returns:
            Proof tree or None if proof failed
        """
        if depth > max_depth:
            return None

        proof_tree = ProofTree(goal=goal)

        if self._is_fact(goal):
            proof_tree.is_proven = True
            return proof_tree

        for conclusion, premises in self.knowledge_base.items():
            subst = Unifier.unify(
                (goal.predicate, goal.args),
                (conclusion.predicate, conclusion.args),
            )

            if subst is not None:
                for premise in premises:
                    instantiated = self._instantiate(premise, subst)
                    sub_proof = self.prove(instantiated, depth + 1, max_depth)

                    if sub_proof is not None or self._is_fact(instantiated):
                        proof_tree.add_step(
                            ProofStep(
                                rule="backward_chain",
                                premises=[],
                                conclusion=goal,
                                substitution=subst,
                            )
                        )
                        proof_tree.is_proven = True
                        return proof_tree

        return None

    def _is_fact(self, formula: Formula) -> bool:
        """Check if formula is a fact."""
        if formula.formula_type != FormulaType.ATOM:
            return False
        return (formula.predicate, formula.args) in {
            (f.predicate, f.args) for f in self.knowledge_base.keys()
        }

    def _instantiate(self, formula: Formula, subst: Substitution) -> Formula:
        """Apply substitution to formula."""
        if formula.formula_type == FormulaType.ATOM:
            new_args = tuple(subst.apply(arg) for arg in formula.args)
            return Formula(
                formula_type=FormulaType.ATOM,
                predicate=formula.predicate,
                args=new_args,
            )
        return formula


class ForwardChainer:
    """Forward chaining theorem prover.

    Applies rules to derive new facts.
    """

    def __init__(
        self,
        rules: List[Tuple[Formula, Formula]],
    ):
        """Initialize forward chainer.

        Args:
            rules: List of (antecedent, consequent) rule pairs
        """
        self.rules = rules

    def derive_facts(
        self,
        facts: Set[Formula],
        max_iterations: int = 100,
    ) -> Set[Formula]:
        """Derive new facts using forward chaining.

        Args:
            facts: Initial facts
            max_iterations: Maximum iterations

        Returns:
            Set of all derived facts
        """
        derived = facts.copy()
        changed = True
        iterations = 0

        while changed and iterations < max_iterations:
            changed = False
            iterations += 1

            for antecedent, consequent in self.rules:
                if self._match(antecedent, derived):
                    if consequent not in derived:
                        derived.add(consequent)
                        changed = True

        return derived

    def _match(self, pattern: Formula, facts: Set[Formula]) -> bool:
        """Check if pattern matches any fact."""
        for fact in facts:
            if pattern.formula_type == FormulaType.ATOM:
                if pattern.predicate == fact.predicate:
                    subst = Unifier.unify(pattern.args, fact.args)
                    if subst is not None:
                        return True
        return False


class TPTPReader:
    """Reader for TPTP (THF) format problems.

    Parses first-order logic problems in TPTP format.
    """

    def __init__(self):
        """Initialize TPTP reader."""
        self.premises: List[Formula] = []
        self.conclusion: Optional[Formula] = None

    def parse(self, content: str) -> Tuple[List[Formula], Formula]:
        """Parse TPTP content.

        Args:
            content: TPTP formula string

        Returns:
            Tuple of (premises, conclusion)
        """
        lines = content.split("\n")
        for line in lines:
            line = line.strip()
            if not line or line.startswith("%"):
                continue

            if line.startswith("fof(") or line.startswith("thf("):
                formula = self._parse_formula(line)
                if "conjecture" in line:
                    self.conclusion = formula
                else:
                    self.premises.append(formula)

        return self.premises, self.conclusion

    def _parse_formula(self, s: str) -> Formula:
        """Parse a single formula."""
        s = s.strip()

        if "=" in s and "!" in s:
            return self._parse_universal(s)
        if "=" in s and "?" in s:
            return self._parse_existential(s)

        if "&" in s:
            parts = s.split("&")
            return Formula(
                formula_type=FormulaType.CONJUNCTION,
                subformulas=(
                    self._parse_formula(parts[0]),
                    self._parse_formula(parts[1]),
                ),
            )

        if "|" in s:
            parts = s.split("|")
            return Formula(
                formula_type=FormulaType.DISJUNCTION,
                subformulas=(
                    self._parse_formula(parts[0]),
                    self._parse_formula(parts[1]),
                ),
            )

        if "=>" in s:
            parts = s.split("=>")
            return Formula(
                formula_type=FormulaType.IMPLICATION,
                subformulas=(
                    self._parse_formula(parts[0]),
                    self._parse_formula(parts[1]),
                ),
            )

        return Formula(
            formula_type=FormulaType.ATOM,
            predicate=s.strip(),
            args=(),
        )

    def _parse_universal(self, s: str) -> Formula:
        """Parse universal formula."""
        return Formula(
            formula_type=FormulaType.UNIVERSAL,
            variable="x",
            subformulas=(Formula(formula_type=FormulaType.ATOM, predicate="formula"),),
        )

    def _parse_existential(self, s: str) -> Formula:
        """Parse existential formula."""
        return Formula(
            formula_type=FormulaType.EXISTENTIAL,
            variable="x",
            subformulas=(Formula(formula_type=FormulaType.ATOM, predicate="formula"),),
        )
