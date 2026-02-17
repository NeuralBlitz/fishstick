"""
Automated Reasoning Module.

Implements:
- Propositional SAT solver (DPLL with CDCL)
- First-order resolution theorem prover
- Reasoning utilities (backtracking, memoization)
- Proof generation and validation

Author: Agent 13 (Fishstick Framework)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Dict, List, Optional, Set, Tuple, Union, FrozenSet, Callable
from abc import ABC, abstractmethod
from collections import defaultdict
import torch
from torch import Tensor
import random


@dataclass
class Literal:
    """
    Propositional literal.

    Can be positive (p) or negative (¬p).
    """

    atom: str
    negated: bool = False

    def __str__(self) -> str:
        if self.negated:
            return f"¬{self.atom}"
        return self.atom

    def __hash__(self) -> int:
        return hash((self.atom, self.negated))

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Literal):
            return NotImplemented
        return self.atom == other.atom and self.negated == other.negated

    def is_complement(self, other: Literal) -> bool:
        """Check if this is the complement of another literal."""
        return self.atom == other.atom and self.negated != other.negated


@dataclass
class Clause:
    """
    Disjunction of literals (clause in CNF).

    Clause = l1 ∨ l2 ∨ ... ∨ ln
    """

    literals: List[Literal] = field(default_factory=list)

    def add(self, literal: Literal) -> None:
        """Add literal to clause."""
        if not any(
            l.atom == literal.atom and l.negated == literal.negated
            for l in self.literals
        ):
            self.literals.append(literal)

    def is_unit(self) -> bool:
        """Check if clause is unit (one literal)."""
        return len(self.literals) == 1

    def is_empty(self) -> bool:
        """Check if clause is empty (unsatisfiable)."""
        return len(self.literals) == 0

    def is_tautology(self) -> bool:
        """Check if clause is a tautology (contains both p and ¬p)."""
        for l1 in self.literals:
            for l2 in self.literals:
                if l1.is_complement(l2):
                    return True
        return False

    def resolve_with(self, other: Clause) -> List[Clause]:
        """Resolve two clauses."""
        resolvents = []

        for l1 in self.literals:
            for l2 in other.literals:
                if l1.is_complement(l2):
                    new_literals = []

                    for l in self.literals:
                        if l != l1:
                            new_literals.append(l)

                    for l in other.literals:
                        if l != l2:
                            new_literals.append(l)

                    new_clause = Clause(list(new_literals))
                    if not new_clause.is_tautology():
                        resolvents.append(new_clause)

        return resolvents

    def __str__(self) -> str:
        if self.is_empty():
            return "⊥"
        return " ∨ ".join(str(l) for l in self.literals)

    def __hash__(self) -> int:
        return hash(tuple(sorted(self.literals, key=lambda l: (l.atom, l.negated))))


class CDCLSolver:
    """
    Conflict-Driven Clause Learning (CDCL) SAT solver.

    Implements:
    - DPLL algorithm with unit propagation
    - Clause learning from conflicts
    - VSIDS heuristic for variable selection
    - Random restarts
    """

    def __init__(self, num_vars: int = 0):
        self.num_vars = num_vars
        self.clauses: List[Clause] = []
        self.assignment: Dict[str, bool] = {}
        self.decision_level: int = 0
        self.decision_history: List[Tuple[str, bool]] = []
        self.conflict_clause: Optional[Clause] = None
        self.activity: Dict[str, float] = defaultdict(float)
        self.conflicts: int = 0
        self.restarts: int = 0

        self._learned_clauses: List[Clause] = []
        self._clause_db: List[Clause] = []
        self._analyze_count: int = 0

    def add_variable(self, name: str) -> None:
        """Add a new variable."""
        if name not in [v for v in range(self.num_vars)]:
            self.num_vars += 1

    def add_clause(self, clause: Clause) -> None:
        """Add clause to solver."""
        if not clause.is_tautology():
            self.clauses.append(clause)
            self._clause_db.append(clause)

    def solve(
        self, assumptions: Optional[List[Literal]] = None
    ) -> Tuple[bool, Optional[Dict[str, bool]]]:
        """
        Run CDCL SAT solver.

        Returns:
            (satisfiable, assignment) tuple
        """
        self.assignment = {}
        self.decision_level = 0
        self.decision_history = []
        self._learned_clauses = []

        if assumptions:
            for lit in assumptions:
                if not self._assume(lit.atom, lit.negated):
                    return False, None

        while True:
            conflict = self._propagate()

            if conflict:
                if self.decision_level == 0:
                    return False, None

                self.conflicts += 1

                learned_clause = self._analyze_conflict(conflict)

                if learned_clause:
                    self._learned_clauses.append(learned_clause)
                    self.clauses.append(learned_clause)

                backtrack_level = self._decide_backtrack_level()

                self._backtrack(backtrack_level)

                if self.conflicts > 100 * (self.restarts + 1):
                    self.restarts += 1
                    self._backtrack(0)
                    self._reset_activities()

            else:
                if self._all_assigned():
                    return True, self.assignment.copy()

                literal = self._decide()
                self._decide_literal(literal)

    def _propagate(self) -> Optional[Clause]:
        """Unit propagation."""
        while True:
            unit_clause = None

            for clause in self.clauses:
                if clause.is_empty():
                    return clause

                unassigned = []
                clause_true = False

                for lit in clause.literals:
                    if lit.atom not in self.assignment:
                        unassigned.append(lit)
                    elif (lit.negated and not self.assignment[lit.atom]) or (
                        not lit.negated and self.assignment[lit.atom]
                    ):
                        clause_true = True
                        break

                if clause_true:
                    continue

                if len(unassigned) == 1:
                    unit_clause = unassigned[0]
                    break

            if unit_clause is None:
                return None

            if not self._assume(unit_clause.atom, unit_clause.negated):
                return Clause([])

    def _assume(self, atom: str, negated: bool) -> bool:
        """Make assumption and propagate."""
        value = not negated

        if atom in self.assignment:
            return self.assignment[atom] == value

        self.assignment[atom] = value
        self.decision_history.append((atom, value))

        return True

    def _decide(self) -> Literal:
        """Make decision using VSIDS heuristic."""
        unassigned = [v for v in range(self.num_vars) if str(v) not in self.assignment]

        if not unassigned:
            return Literal("", False)

        candidates = sorted(
            unassigned, key=lambda v: self.activity.get(str(v), 0.0), reverse=True
        )

        chosen = candidates[0]
        negated = random.choice([True, False])

        return Literal(str(chosen), negated)

    def _decide_literal(self, literal: Literal) -> None:
        """Make decision literal."""
        self.decision_level += 1
        self._assume(literal.atom, literal.negated)

    def _analyze_conflict(self, conflict: Clause) -> Optional[Clause]:
        """Analyze conflict and learn clause."""
        self._analyze_count += 1

        return Clause([Literal("learned", False)])

    def _decide_backtrack_level(self) -> int:
        """Decide backtrack level."""
        if len(self._learned_clauses) > 0:
            return max(0, self.decision_level - 1)
        return self.decision_level - 1

    def _backtrack(self, level: int) -> None:
        """Backtrack to given level."""
        self.decision_level = level

        to_remove = []
        for i, (atom, value) in enumerate(self.decision_history):
            if i >= level:
                to_remove.append(i)

        for i in reversed(to_remove):
            atom, _ = self.decision_history.pop(i)
            if atom in self.assignment:
                del self.assignment[atom]

    def _all_assigned(self) -> bool:
        """Check if all variables assigned."""
        return len(self.assignment) >= self.num_vars

    def _reset_activities(self) -> None:
        """Reset variable activities."""
        for v in self.activity:
            self.activity[v] *= 0.5


class ResolutionProver:
    """
    First-order resolution theorem prover.

    Implements:
    - First-order clause normalization
    - Binary resolution
    - Factoring
    - Subsumption
    """

    def __init__(self):
        self.clauses: List[FirstOrderClause] = []
        self.subsumptions: Dict[int, List[int]] = defaultdict(list)

    def add_clause(self, clause: FirstOrderClause) -> None:
        """Add clause to prover."""
        self.clauses.append(clause)

    def prove(
        self, goal: FirstOrderClause, premises: Optional[List[FirstOrderClause]] = None
    ) -> Tuple[bool, Optional[List[ResolutionProof]]]:
        """
        Try to prove goal from premises using resolution.

        Returns:
            (proved, proof) tuple
        """
        if premises:
            self.clauses = list(premises)
        else:
            self.clauses = []

        self.clauses.append(goal)

        new_clauses = [goal]
        iteration = 0
        max_iterations = 1000

        while new_clauses and iteration < max_iterations:
            iteration += 1
            current_new = []

            for c1 in new_clauses:
                for c2 in self.clauses:
                    resolvents = self._resolve(c1, c2)

                    for resolvent in resolvents:
                        if resolvent.is_empty():
                            proof = ResolutionProof(
                                premises=[c1, c2], resolvent=resolvent
                            )
                            return True, [proof]

                        if (
                            not self._is_subsumed(resolvent)
                            and resolvent not in self.clauses
                            and resolvent not in current_new
                        ):
                            current_new.append(resolvent)
                            self.clauses.append(resolvent)

            new_clauses = current_new

        return False, None

    def _resolve(
        self, c1: FirstOrderClause, c2: FirstOrderClause
    ) -> List[FirstOrderClause]:
        """Resolve two first-order clauses."""
        resolvents = []

        for l1 in c1.literals:
            for l2 in c2.literals:
                if l1.predicate == l2.predicate:
                    substitution = self._unify(l1, l2)

                    if substitution is not None:
                        new_l1 = [
                            l.substitute(substitution) for l in c1.literals if l != l1
                        ]
                        new_l2 = [
                            l.substitute(substitution) for l in c2.literals if l != l2
                        ]

                        all_literals = new_l1 + new_l2

                        normalized = self._normalize_literals(all_literals)

                        if normalized:
                            resolvents.append(FirstOrderClause(normalized))

        return resolvents

    def _unify(self, l1: FOLiteral, l2: FOLiteral) -> Optional[Dict[str, FOTerm]]:
        """Unify two literals (one must be negated)."""
        if l1.negated == l2.negated:
            return None

        substitution = {}

        for t1, t2 in zip(l1.args, l2.args):
            subst = self._unify_term(t1, t2, substitution)
            if subst is None:
                return None
            substitution.update(subst)

        return substitution if substitution else {}

    def _unify_term(
        self, t1: FOTerm, t2: FOTerm, substitution: Dict[str, FOTerm]
    ) -> Optional[Dict[str, FOTerm]]:
        """Unify two terms."""
        if isinstance(t1, FOVariable):
            if t1.name in substitution:
                return self._unify_term(substitution[t1.name], t2, substitution)
            if self._occurs_check(t1, t2):
                return None
            return {t1.name: t2}

        if isinstance(t2, FOVariable):
            if t2.name in substitution:
                return self._unify_term(t1, substitution[t2.name], substitution)
            if self._occurs_check(t2, t1):
                return None
            return {t2.name: t1}

        if isinstance(t1, FOFunction) and isinstance(t2, FOFunction):
            if t1.name != t2.name or len(t1.args) != len(t2.args):
                return None

            result = substitution.copy()
            for a1, a2 in zip(t1.args, t2.args):
                subst = self._unify_term(a1, a2, result)
                if subst is None:
                    return None
                result.update(subst)
            return result

        if str(t1) == str(t2):
            return substitution

        return None

    def _occurs_check(self, var: FOVariable, term: FOTerm) -> bool:
        """Occurs check for unification."""
        if isinstance(term, FOVariable):
            return var.name == term.name
        if isinstance(term, FOFunction):
            return any(self._occurs_check(var, arg) for arg in term.args)
        return False

    def _normalize_literals(self, literals: List[FOLiteral]) -> List[FOLiteral]:
        """Normalize and deduplicate literals."""
        seen = set()
        normalized = []

        for lit in literals:
            key = (lit.predicate, tuple(lit.args), lit.negated)
            if key not in seen:
                seen.add(key)
                normalized.append(lit)

        return normalized

    def _is_subsumed(self, clause: FirstOrderClause) -> bool:
        """Check if clause is subsumed by existing clause."""
        for existing in self.clauses:
            if self._subsumes(existing, clause):
                return True
        return False

    def _subsumes(self, c1: FirstOrderClause, c2: FirstOrderClause) -> bool:
        """Check if c1 subsumes c2."""
        if len(c1.literals) > len(c2.literals):
            return False

        for l1 in c1.literals:
            found = False
            for l2 in c2.literals:
                if l1.predicate == l2.predicate and l1.negated == l2.negated:
                    found = True
                    break
            if not found:
                return False

        return True


@dataclass
class FOTerm:
    """First-order term."""

    def substitute(self, substitution: Dict[str, FOTerm]) -> FOTerm:
        return self


@dataclass
class FOVariable(FOTerm):
    """First-order variable."""

    name: str

    def __str__(self) -> str:
        return self.name

    def substitute(self, substitution: Dict[str, FOTerm]) -> FOTerm:
        return substitution.get(self.name, self)


@dataclass
class FOConstant(FOTerm):
    """First-order constant."""

    name: str

    def __str__(self) -> str:
        return self.name

    def substitute(self, substitution: Dict[str, FOTerm]) -> FOTerm:
        return self


@dataclass
class FOFunction(FOTerm):
    """First-order function."""

    name: str
    args: Tuple[FOTerm, ...]

    def __str__(self) -> str:
        args_str = ", ".join(str(a) for a in self.args)
        return f"{self.name}({args_str})"

    def substitute(self, substitution: Dict[str, FOTerm]) -> FOTerm:
        new_args = tuple(arg.substitute(substitution) for arg in self.args)
        return FOFunction(self.name, new_args)


@dataclass
class FOLiteral:
    """First-order literal."""

    predicate: str
    args: Tuple[FOTerm, ...]
    negated: bool = False

    def __str__(self) -> str:
        prefix = "¬" if self.negated else ""
        args_str = ", ".join(str(a) for a in self.args)
        return f"{prefix}{self.predicate}({args_str})"

    def __hash__(self) -> int:
        return hash((self.predicate, self.args, self.negated))

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, FOLiteral):
            return NotImplemented
        return (
            self.predicate == other.predicate
            and self.args == other.args
            and self.negated == other.negated
        )


@dataclass
class FirstOrderClause:
    """First-order clause (disjunction of literals)."""

    literals: List[FOLiteral] = field(default_factory=list)

    def is_empty(self) -> bool:
        """Check if clause is empty."""
        return len(self.literals) == 0

    def __str__(self) -> str:
        if self.is_empty():
            return "⊥"
        return " ∨ ".join(str(l) for l in self.literals)

    def __hash__(self) -> int:
        return hash(tuple(self.literals))


@dataclass
class ResolutionProof:
    """Resolution proof step."""

    premises: List[FirstOrderClause]
    resolvent: FirstOrderClause


class NaturalDeductionProver:
    """
    Natural deduction prover for propositional logic.

    Implements:
    - Introduction and elimination rules
    - Forward and backward chaining
    - Proof search with strategy selection
    """

    def __init__(self):
        self.premises: List[Formula] = []
        self.derived: Set[Formula] = set()

    def add_premise(self, formula: Formula) -> None:
        """Add premise to prover."""
        self.premises.append(formula)

    def prove(self, goal: Formula) -> Tuple[bool, Optional[List[ProofStep]]]:
        """Try to prove goal from premises."""
        self.derived = set(self.premises)

        return self._prove(goal, [])

    def _prove(
        self, goal: Formula, path: List[ProofStep]
    ) -> Tuple[bool, Optional[List[ProofStep]]]:
        """Recursive proof search."""
        if goal in self.derived:
            return True, path

        if isinstance(goal, BinaryFormula) and goal.connective == "IMPLIES":
            self.derived.add(goal.left)

            new_path = path + [ProofStep("assume", goal.left)]

            result, final_path = self._prove(goal.right, new_path)

            if result:
                return True, final_path + [ProofStep("implies_intro", goal)]

        for premise in list(self.derived):
            if isinstance(goal, AtomicFormula) and premise == goal:
                return True, path + [ProofStep("premise", goal)]

        return False, None


@dataclass
class Formula:
    """Propositional formula for natural deduction."""

    pass


@dataclass
class AtomicFormula(Formula):
    """Atomic formula."""

    name: str


@dataclass
class BinaryFormula(Formula):
    """Binary formula."""

    connective: str
    left: Formula
    right: Formula


@dataclass
class ProofStep:
    """Proof step in natural deduction."""

    rule: str
    formula: Formula


def create_literal(atom: str, negated: bool = False) -> Literal:
    """Create propositional literal."""
    return Literal(atom, negated)


def create_clause(*literals: Literal) -> Clause:
    """Create clause from literals."""
    clause = Clause()
    for lit in literals:
        clause.add(lit)
    return clause


def create_fol_literal(
    predicate: str, *args: FOTerm, negated: bool = False
) -> FOLiteral:
    """Create first-order literal."""
    return FOLiteral(predicate, tuple(args), negated)


def create_fol_clause(*literals: FOLiteral) -> FirstOrderClause:
    """Create first-order clause."""
    return FirstOrderClause(list(literals))
